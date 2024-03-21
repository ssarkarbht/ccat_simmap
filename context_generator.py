#!/bin/python

'''
CCAT Collaboration (Sourav Sarkar),
March 17, 2024

Description: This script is prepared to generate database and context
    file from a TOAST simulation dumped into HDF5/G3 files, as the db and
    context are required for utilizing several sotodlib tools for
    mapmaking and post processing.
    (Modified and adapted from SO's `write_context.py` script)

'''

#import stuff
from sotodlib.core import metadata
from so3g.proj import quat
from sotodlib.io.metadata import write_dataset, read_dataset
import sotodlib.toast as sotoast

from spt3g import core as g3

import numpy as np
import h5py as h5
import yaml

from glob import glob
import os, sys, time

#rad conversion
DEG = np.pi/180.0

#tube_types (NOTE: Need updates)
tube_types = {'f280': 'CMBPol'}

class ContextGenerator:
    '''Base class for creating context file generator from either
    hdf5 or g3 file format of the simulation output.
    '''
    def __init__(self, base_dir, obs_prefix, file_ext = ".h5",
            tel_name='FYST'):
        '''Args:
        base_dir (str): High-level directory path containing the
                        tod data files
        obs_prefix (str): Common prefix in the case where each
                    observation is stored into separate directories
        file_ext (str): Extension of the file types being loaded.
        tel_name (str): Main telescope name.
        '''
        #get all the filenames (per observation)
        self.filelist = []
        if obs_prefix is not None:
            dirs = sorted(glob(base_dir + obs_prefix + "*"))
            for d in dirs:
                curr_files = sorted(glob(d + "*" + file_ext))
                self.filelist += curr_files
        elif obs_prefix is None:
            curr_files = sorted(glob(base_dir + "*" + file_ext))
            self.filelist += curr_files
        print (f"Loaded {len(self.filelist)} observation files.")

        self.tel_name = tel_name

    def load_detinfo(self, fname=None):
        '''Modify this function in the subclass to return the telescope
        name and focal plane info, which are then used in the detdb
        creation.
        '''
        pass

    def create_detdb(self, fname=None, db=None):
        '''Extract detector information from the frames
        and create detdb.
        Args:
            fname (str): Name of the input file for extracting
                    detector info. Default is None, where the
                    first file in the list is used and detector
                    info is assumed to be the same across all
                    observation.
            db (database): Any existing database for update.
                Default is None, so create it from scratch.
        '''
        if db is None:
            db = metadata.DetDb()
            #make the base table
            db.create_table('base',
                    ["`det_id_` text",
                    "`readout_id` text",
                    "`wafer_slot` text",
                    "`special_ID` text",
                    "`tel_type` text",
                    "`tube_type` text",
                    "`band` text",
                    "`fcode` text",
                    "`toast_band` text",
                        ])

            #make the quat table
            db.create_table('quat',
                    ["`r` float",
                    "`i` float",
                    "`j` float",
                    "`k` float"
                        ])
        #get the new info from the load function
        self.tel_type, self.fp = self.load_detinfo(fname=fname)

        #get any existing detector info
        existing = list(db.dets()['name'])

        #get the required focal plane detector info
        for dv in self.fp:
            v = dict([(_k, dv[_k].decode('ascii'))
                for _k in ['wafer_slot', 'band', 'name']])
            k = v.pop('name')
            #if the current detector already exist, ignore
            if k in existing:
                continue
            v['special_ID'] = int(dv['uid'])
            v['toast_band'] = v['band']
            v['band'] = v['toast_band'].split('_')[1]
            v['fcode'] = v['band']
            v['tel_type'] = self.tel_type
            v['tube_type'] = tube_types[v['band']]
            v['det_id_'] = 'DET_' + k
            v['readout_id'] = k
            db.add_props('base', k, **v, commit=False)
            db.add_props('quat', k, **{'r': dv['quat'][3],
                                       'i': dv['quat'][0],
                                       'j': dv['quat'][1],
                                       'k': dv['quat'][2]})

        #prepare and validate db
        db.conn.commit()
        db.validate()
        return db

    def detdb_to_focalplane(self, db):
        '''Converts the focal plane info to compatible format,
        like, plannet mapper
        Args:
            db (database): detdb to get the info from.
        Returns:
            Converted focal plane variables (metadata.resultset)
        '''
        fp = metadata.ResultSet(keys=['dets:readout_id',
                                      'xi', 'eta', 'gamma'])
        for row in db.props(props=['readout_id', 'quat.r',
                                'quat.i', 'quat.j', 'quat.k']).rows:
            q = quat.quat(*row[1:])
            xi, eta, gamma = quat.decompose_xieta(q)
            fp.rows.append((row[0], xi, eta, (gamma) % (2*np.pi)))

        return fp

    def load_obsinfo(self, fname):
        '''Modify this in the subclass to load the observation data
        from the given single observation file accordingly.
        '''
        pass
    
    def create_obsdb(self, fname, db=None, **kwargs):
        '''For each observation file, create an entry in the obsdb
        and prepare, validate the full db.
        Args:
            fname (str): Name of the input file for extracting obsinfo
            db (database): Provide existing database object in
                    update-only case.
        Returns:
            obs_id: A unique observation ID
            db: Updated database
        '''
        if db is None:
            db = metadata.ObsDb()
            db.add_obs_columns([#Standardized
                            'timestamp float',
                            'duration float',
                            'start_time float',
                            'stop_time float',
                            'type string',
                            'subtype string',
                            'telescope string',
                            'telescope_flavor string',
                            'tube_slot string',
                            'tube_flavor string',
                            'detector_flavor string',
                            
                            #soon-to-be standardized
                            'wafer_slot_mask string',
                            'el_nom float',
                            'el_span float',
                            'az_nom float',
                            'az_span float',
                            'roll_nom float',
                            'roll_span float',

                            #Extensions (simulation or otherwise)
                            'wafer_slots string',
                            'type string',
                            'target string',
                            'toast_obs_name string',
                            'toast_obs_uid string',

                            ])

        #get the observation info for the current file
        obs_info = self.load_obsinfo(fname)

        #add any extra bs info passed through kwargs
        obs_info.update(**kwargs)

        #create the observation id
        obs_id = f'{int(obs_info["timestamp"])}_{obs_info["tube_slot"]}'

        #update db
        db.update_obs(obs_id, obs_info)
        print (f"++++++ Added observation: {obs_id}")

        return obs_id, db

    def process_obs(self, context_dir='context',
            det_per_obs=False):
        '''Wrapper function to process all the observation files and
        create the complete database and context file.
        Args:
            context_dir (str): Path to the head dir where the databases
                        and the context file will be stored.
            det_per_obs (bool): Whether to extract detector information
                        for every observation separately.
        '''
        detsets = {}
        self.obsdb = None
        self.obsfiledb = metadata.ObsFileDb()

        if not(det_per_obs):
            self.detdb = self.create_detdb()
            self.props = self.detdb.props()
            self.props.keys[self.props.keys.index('det_id_')] = 'det_id'
            
        for f in self.filelist:
            print (f"Processing file: {f}")
            if det_per_obs:
                self.detdb = self.create_detdb()
                self.props = self.detdb.props()
                self.props.keys[self.props.keys.index('det_id_')] = 'det_id'

            #TODO: This is hardcoded, need to update once hk is finalized
            extra_obsinfo = {'telescope': self.tel_type,
                            'telescope_flavor': 'CMBPol',
                            'detector_flavor': 'TES',
                            'tube_slot': 'f280',
                            'tube_flavor': self.props['tube_type'][0],
                            'wafer_slot_mask': '_',
                            'wafer_slots': 'w12'}

            obs_id, self.obsdb = self.create_obsdb(f, 
                                           db=self.obsdb, **extra_obsinfo)
            detset = '_'.join([self.props['wafer_slot'][0],
                                self.props['band'][0]])
            if detset not in detsets:
                fp = self.detdb_to_focalplane(self.detdb)
                detsets[detset] = [self.props, fp]
                self.obsfiledb.add_detset(detset, 
                                        self.props['readout_id'])

            self.obsfiledb.add_obsfile(f, obs_id, detset, 0, 1)

        #Finally, write everything
        if not os.path.exists(context_dir):
            os.makedirs(context_dir, exist_ok=True)

        self.obsdb.to_file(f'{context_dir}/obsdb.sqlite')
        self.obsfiledb.to_file(f'{context_dir}/obsfiledb.sqlite')

        #create metadata instead of detdb
        scheme = metadata.ManifestScheme()
        scheme.add_exact_match('dets:detset')
        scheme.add_data_field('dataset')
        db1 = metadata.ManifestDb(scheme=scheme)

        scheme = metadata.ManifestScheme()
        scheme.add_exact_match('dets:detset')
        scheme.add_data_field('dataset')
        db2 = metadata.ManifestDb(scheme=scheme)

        for detset, (props, fp) in detsets.items():
            key = 'dets_' + detset
            props.keys = ['dets:' + k for k in props.keys]
            write_dataset(props, f'{context_dir}/metadata.h5', 
                        key, overwrite=True)
            db1.add_entry({'dets:detset': detset, 'dataset': key},
                            filename='metadata.h5')

            key = 'focalplane_' + detset
            write_dataset(fp, f'{context_dir}/metadata.h5', 
                        key, overwrite=True)
            db2.add_entry({'dets:detset': detset, 'dataset': key},
                            filename='metadata.h5')

        db1.to_file(f'{context_dir}/det_info.sqlite')
        db2.to_file(f'{context_dir}/focalplane.sqlite')

        #context.yaml
        context = {
                'tags': {'metadata_lib': './'},
                'imports': ['sotodlib.io.metadata'],
                'obsfiledb': '{metadata_lib}/obsfiledb.sqlite',
                #'detdb': '{metadata_lib}/detdb.sqlite',
                'obsdb': '{metadata_lib}/obsdb.sqlite',
                'obs_loader_type': 'toast3-hdf',
                'obs_colon_tags': ['wafer_slot', 'band'],
                'metadata': [
                    {'db': "{metadata_lib}/det_info.sqlite",
                        'det_info': True},
                    {'db': "{metadata_lib}/focalplane.sqlite",
                        'name': "focal_plane"}]
                }
        open(f'{context_dir}/context.yaml', 'w').write(yaml.dump(context,
                                                        sort_keys=False))
        return None

class H5ContextWriter(ContextGenerator):
    '''Inherits from context generator class to adapt to loading
    HDF5 files.
    '''
    def __init__(self, base_dir, **kwargs):
        super().__init__(base_dir, None, file_ext='.h5', **kwargs)

    def load_detinfo(self, fname=None):
        if fname is None:
            fname = self.filelist[0]
        hf = h5.File(fname, 'r')

        tel_type = hf['instrument'].attrs.get('telescope_name')
        fp = np.array(hf['instrument']['focalplane'])
        hf.close()
        return tel_type, fp

    def load_obsinfo(self, fname):
        h = h5.File(fname, 'r')
        t = np.asarray(h['shared']['times'])[[0,-1]]
        az = np.asarray(h['shared']['azimuth'][()])
        el = np.asarray(h['shared']['elevation'][()])
        el_nom = (el.max() + el.min()) / 2
        el_span = el.max() - el.min()

        az_cut = az[0] - np.pi
        az = (az - az_cut) % (2 * np.pi) + az_cut
        az_span = az.max() - az.min()
        az_nom = (az.max() + az.min()) / 2 % (2 * np.pi)

        data = {
                'toast_obs_name': h.attrs['observation_name'],
                'toast_obs_uid': int(h.attrs['observation_uid']),
                'target': h.attrs['observation_name'].split('-')[0].lower(),
                'start_time': t[0],
                'stop_time': t[1],
                'timestamp': t[0],
                'duration': t[1] - t[0],
                'type': 'obs',
                'subtype': 'survey',
                'el_nom': el_nom / DEG,
                'el_span': el_span / DEG,
                'az_nom': az_nom / DEG,
                'az_span': az_span / DEG,
                'roll_nom': 0.,
                'roll_span': 0.,
                }
        h.close()

        return data
