from __future__ import print_function

import numpy as np
import calcGrid
import os.path as path
import pylab
import matplotlib
import struct
import time
import h5py
import scipy.linalg
import loaders
import sys
from const import *
from math import *
from utilities import *
from cosmological_factors import *
import sys
import os

class gadget_snapshot(DirMixIn):
    hdf5_name_conversion = {
        'Coordinates':'pos',
        'Velocities':'vel',
        'BirthPos':'bpos',
        'BirthVel':'bvel',
        'BirthDensity':'bdens',
        'DM_VoronoiDensity':'vode',
        'TracerID':'trid',
        'ParentID':'prid',
        'FluidQuantities':'fldq',
        'InternalEnergy':'u',
        'ParticleIDs' :'id',
        'Masses' : 'mass',
        'Density' :'rho',
        'Volume' : 'vol',
        'Pressure' : 'pres',
        'SmoothingLength' : 'hsml',
        'Nuclear Composition' : 'xnuc',
        'NuclearComposition' : 'xnuc',
        'Passive Scalars' : 'pass',
        'PassiveScalars' : 'pass',
        'StarFormationRate' :'sfr',
        'StellarFormationTime' : 'age',
        'FeedbackDone' : 'nfb',
        'GFM_StellarFormationTime' : 'age',
        'GFM_InitialMass' : 'gima',
        'GFM_Metallicity' : 'gz',
        'GFM_MassReleased' : 'gimr',
        'GFM_MetalReleased' : 'gimz',
        'GFM_MetalsReleased' : 'gmmz',
        'GFM_StellarPhotometrics' : 'gsph',
        'GFM_AGNRadiation' : 'agnr',
        'GFM_CoolingRate' : 'gcol',
        'GFM_Metallicity' : 'gz',
        'GFM_Metals' : 'gmet',
        'GFM_RProcess' : 'gmrp',
        'GFM_NSNS_Count' : 'gmrc',
        'GravPotential' : 'gpot',
        'Metallicity' : 'z',
        'Potential' :'pot',
        'GravPotential' : 'pot',
        'Acceleration' : 'acce',
        'PM Acceleration' : 'pmac',
        'Tree Acceleration' : 'trac',
        'TimeStep' : 'tstp',
        'MagneticField' : 'bfld',
        'MagneticFieldPsi' : 'psi',
        'DednerSpeed' : 'vded',
        'CurlB' : 'curb',
        'DivBCleening' : 'psi',
        'SmoothedMagneticField' : 'bfsm',
        'RateOfChangeOfMagneticField' : 'dbdt',
        'DivergenceOfMagneticField' : 'divb',
        'MagneticFieldDivergenceAlternative' : 'dvba',
        'MagneticFieldDivergence' : 'divb',
        'PressureGradient' : 'grap',
        'DensityGradient' : 'grar',
        'BfieldGradient' : 'grab',
        'VelocityGradient' : 'grav',
        'Center-of-Mass' : 'cmce',
        'CenterOfMass' : 'cmce',
        'Surface Area' : 'area',
        'Number of faces of cell' : 'nfac',
        'VertexVelocity' : 'veve',
        'Divergence of VertexVelocity' : 'divv',
        'VelocityDivergence' : 'divv',
        'Temperature' : 'temp',
        'Vorticity' : 'vort',
        'AllowRefinement' : 'ref',
        'ElectronAbundance' : 'ne',
        'HighResGasMass' : 'hrgm',
        'NeutralHydrogenAbundance': 'nh',
        'Star Index': 'sidx',
        'CellSpin': 'spin',
        'Spin Center': 'lpos',
        'Machnumber' :'mach',
        'EnergyDissipation' : 'edis',
        'CosmicRaySpecificEnergy': 'cren',
        'CosmicRayStreamingRegularization': 'chi',
        'CRPressureGradient': 'crpg',
        'BfieldGradient': 'bfgr',
        'Nucler energy generation rate': 'dedt',
        'NuclearEnergyGenerationRate': 'dedt',
        'MolecularWeight' : 'mu',
        'SoundSpeed' : 'csnd',
        'Jet_Tracer' : 'jetr',

        'Softenings': 'soft',
        'GravityInteractions' : 'gint',
        'SFProbability' : 'sfpr',
        'TurbulentEnergy' : 'utur',
        'AccretedFlag':'accf',
        'Erad' : 'erad',
        'Lambda' : 'lamd',
        'Kappa_P' : 'ka_p',
        'Kappa_R' : 'ka_r',
        'graderad' : 'grae',

        'BH_CumMassGrowth_QM' : 'bcmq',
        'BH_CumMassGrowth_RM' : 'bcmr',
        'BH_Hsml' : 'bhhs',
        'BH_Pressure' : 'bhpr',
        'BH_U' : 'bhu',
        'BH_Density' : 'bhro',
        'BH_Mdot' : 'bhmd',
        'BH_Mdot_Radio' : 'bhmr', 
        'BH_Mdot_Quasar' : 'bhmq', 
        'BH_Mass' : 'bhma', 
        'BH_MassMetals' : 'bhmz', 
        'BH_TimeStep' : 'bhts', 
        'BH_MdotBondi' : 'bhbo',
        'BH_MdotEddington' : 'bhed',

        'ChemicalAbundances' : 'chma',
        ' Dust Temperature' : 'dtem',
        'CoolTime' : 'cltm',
        'MagneticVectorPotential' : 'afld',
        'MagneticAShiftX' : 'ashx',
        'MagneticAShiftY' : 'ashy',
        'MagneticAShiftZ' : 'ashz',

        'WindTrackerMassFraction' : 'gtwi',
        'SNTrackerMassFraction' : 'gtsn',
        'TimebinHydro' : 'tbhd',

        'AGNFlag' : 'flag', 
        'CoolingRate' : 'coor',
        'AGNWindFraction' : 'wind'
    }

    hdf5_part_field = {
        'pos' : 'all',
        'vel' : 'all',
        'bpos' : 'stars',
        'bvel' : 'stars',
        'bdens': 'stars',
        'vode' : 'dm',
        'trid' : 'tracers',
        'prid' : 'tracers',
        'fldq' : 'tracers',
        'u' : 'gas',
        'id' : 'all',
        'mass' : 'all',
        'rho' : 'gas',
        'vol' : 'gas',
        'pres' : 'gas',
        'hsml' : 'gas',
        'xnuc' : 'gas',
        'pass' : 'gas',
        'sfr' : 'gas',
        'age' : 'stars',
        'nfb' : 'stars',  #CHECK!!!
        'gima' : 'stars',
        'gimr' : 'stars',
        'gimz' : 'stars',
        'gmmz' : 'stars',
        'accf' : 'stars',
        'gz' : 'baryons',
        'gsph' : 'stars',
        'gpot' : 'stars',
        'agnr' : 'gas',
        'gcol' : 'gas',
        'gmet' : 'baryons',
        'gmrp' : 'baryons',
        'gmrc' : 'baryons',
        'z' : 'baryons', #CHECK!!
        'pot' : 'all',
        'acce' : 'all',
        'pmac' : 'all',
        'trac' : 'all',
        'tstp' : 'all',
        'bfld' : 'gas',
        'curb' : 'gas',
        'afld' : 'gas',
        'ashx' : 'gas',
        'ashy' : 'gas',
        'ashz' : 'gas',
        'psi' : 'gas',
        'vded' : 'gas',
        'bfsm' : 'gas',
        'dbdt' : 'gas',
        'divb' : 'gas',
        'dvba' : 'gas',
        'grap' : 'gas',
        'grar' : 'gas',
        'grab' : 'gas',
        'grav' : 'gas',
        'cmce' : 'gas',
        'area' : 'gas',
        'nfac' : 'gas',
        'veve' : 'gas',
        'divv' : 'gas',
        'divv' : 'gas',
        'temp' : 'gas',
        'vort' : 'gas',
        'ref' : 'gas',
        'xnuc' : 'gas',
        'ne' : 'gas',
        'hrgm' : 'gas',
        'nh' : 'gas',
        'sidx' : 'gas',  
        'spin' : 'gas',
        'lpos' : 'gas',
        'mach' : 'gas',
        'edis' : 'gas',
        'cren' : 'gas',
        'chi'  : 'gas',
        'dedt' : 'gas',
        'bfgr' : 'gas',
        'crpg' : 'gas',
        'mu'   : 'gas',
        'csnd' : 'gas',
        'jetr' : 'gas',

        'soft' : 'all',
        'gint' : 'all',
        'sfpr' : 'gas',
        'utur' : 'gas',

        'bcmq' : 'bh',
        'bcmr' : 'bh',
        'bhhs' : 'bh',
        'bhpr' : 'bh',
        'bhu' : 'bh',
        'bhro' : 'bh',
        'bhmd' : 'bh',
        'bhmr' : 'bh', 
        'bhmq' : 'bh', 
        'bhma' : 'bh',
        'bhmz' : 'bh',
        'bhts' : 'bh', 
        'bhbo' : 'bh',
        'bhed' : 'bh',

        'chma' : 'gas',
        'dtem' : 'gas',
        'cltm' : 'gas',

        'gtsn' : 'baryons',
        'gtwi' : 'baryons',
        'tbhd' : 'gas',

        'flag' : 'gas',
        'coor' : 'gas',
        'wind' : 'gas'
    }

    def __init__( self, filename, verbose=False, chunk=None, loadonly=False, loadonlytype=False, applytransformationfacs=True, onlyHeader=False, nommap=False, tracer=False, hdf5=False, species=False, forcesingleprec=False, cosmological=None, loadonlyhalo=-1, subfind=None, lazy_load=False, forcedoubleprec=False, convert_posvel_to_double=False, quiet=False, convAddit=False, offsets=None ):
        """Main class for loading Gadget and Arepo snapshots.

        Here could be a description of all parameters.

        - lazy_load: set to True to load the data from the files on demand,
            i.e., when s.data[key] or s.key is accessed. s.data.keys() gives
            a list of all available data blocks
        """

        if type( loadonly ) == np.ndarray or type( loadonly ) == list:
            loadlist = loadonly
        else:
            loadlist = []

        self.filename = filename
        self.tracer = tracer
        self.hdf5 = hdf5
        self.forcesingleprec = forcesingleprec
        self.forcedoubleprec = forcedoubleprec
        self.applytransformationfacs = applytransformationfacs
        self.lazy_load = lazy_load
        self.chunk = chunk
        # ensure self.data is always present
        self.data = {}
        self.transformation_factors = {}
        self.loadonlyhalo = loadonlyhalo
        self.convert_posvel_to_double = convert_posvel_to_double
        self.quiet = quiet
        self.cosmological = cosmological
        self.convAddit = convAddit
        self.offsets = offsets

        if loadonlyhalo >= 0:
            if subfind == None or not 'flty' in subfind.data:
                print( "You need to provide a subfind object that has 'flty' loaded to load only one halo. Loading full snapshot." )
                loadonlyhalo = -1
            else:
                self.subfind = subfind

        if type( loadonlytype ) == np.ndarray or type( loadonlytype ) == list:
            if not self.hdf5:
                print( "loadonlytype is not supported for formats different from hdf5. Loading all particle types" )
                self.loadtype = [0, 1, 2, 3, 4, 5]
            else:
                self.loadtype = loadonlytype
        else:
            self.loadtype = [0, 1, 2, 3, 4, 5]

        suffix = ""
        # check for extension; switch to HDF5 if needed
        if not path.exists( filename ):
            if path.exists( filename + ".hdf5" ) or path.exists( filename + ".0.hdf5" ):
                suffix = ".hdf5"
                self.hdf5 = True
            if path.exists( filename + ".h5" ) or path.exists( filename + ".0.h5" ):
                suffix = ".h5"
                self.hdf5 = True

        self.filecount = 1
        if path.exists( filename + suffix ):
            self.files = [filename + suffix]
        elif path.exists( filename + '.0' + suffix ):
            self.files = [filename + '.0' + suffix]
            while path.exists( filename + ".%d" % self.filecount + suffix):
                self.files += [filename + ".%d" % self.filecount + suffix]
                self.filecount += 1

            if not nommap:
                if verbose:
                    print("Multiple files detected, thus mmap is deactivated.")
                nommap = True
        else:
            raise Exception( "Neither %s nor %s.0 exists." % (filename, filename) )
        self.file_name = filename
        self.nommap = nommap

        self.twodim = False
        self.onedim = False
        self.center = np.array( [0,0,0], dtype="float64" )
        self._center_initialized = False
        
        if self.hdf5:
            self.load_header_hdf5( 0, verbose=verbose, verboselevel=True )
            if onlyHeader:
                return

            if lazy_load:
                # set data to lazy dict, load data only if accessed
                self.get_blocknames_hdf5(verbose)
                self.blocknames.append('vol')
                self.blocknames.append('type')
                self.verbose = verbose
                self.chunk = chunk
                self.loadonlyhalo = loadonlyhalo
                self.data = LazyDict(self, '_lazy_load_data_hdf5',
                    lazy_keys=self.blocknames,
                    vector_keys=self.vector_blocks)
            else:
                self.load_data_hdf5( verbose=verbose, loadlist=loadlist, chunknum=chunk, loadonlyhalo=loadonlyhalo )
        else:
            self.datablocks_skip = ["HEAD", "INFO"]
            self.datablocks_int32 = ["ID  ","NONN"]

            self.load_header( 0, verbose=verbose )
            self.get_blocks( 0, verbose=verbose )

            if onlyHeader:
                return

            if lazy_load:
                # set data to lazy dict, load data only if accessed
                self.get_block_names_type2(verbose)
                self.blocknames.append('vol')
                self.verbose = verbose
                self.data = LazyDict(self, '_lazy_load_data',
                    lazy_keys=self.blocknames,
                    vector_keys=self.vector_blocks)
            else:
                if self.filecount == 1 and not nommap:
                    self.load_data_mmap( verbose=verbose )
                else:
                    self.load_data( verbose=verbose, loadlist=loadlist )

        self.convenience()

        if self.hubbleparam == 0.:
            self.hubbleparam = 1.

        if((self.redshift > 0 and not cosmological == False) or (cosmological == True)):
            self.cosmology_init()
            self.set_transformation_factors_comoving()
        else:
            self.set_transformation_factors_non_comoving()

        if not lazy_load and applytransformationfacs:
            self.apply_transformation_factors()

        if species:
            self.species = loaders.load_species( species )

        if 'pos' in self.data:
            if np.abs( self.data['pos'][:,2] ).max() == 0.:
                self.center[2] = 0.
                if np.abs( self.data['pos'][:,1] ).max() == 0.:
                  self.center[1] = 0.
                  self.onedim = True
                else:
                  self.twodim = True

        # add data arrays for subtypes
        for ptype in self.loadtype:
            setattr(self, 'data_type{:d}'.format(ptype), IndexDict(self, ptype))

    def __getattr__(self, name):
        """enable access via object attributes to data dict entries"""
        if name in self.data:
            return self.data[name]
        if name == "acc" and "acce" in self.data:
            return self.data["acce"]
        if name == "hmass" and "hrgm" in self.data:
            return self.data["hrgm"]
        if name == "vol" and "rho" in self.data and "mass" in self.data:
            self.data["vol"] = self.mass[:self.nparticlesall[0]] / self.rho.astype('f8')
            return self.data["vol"]

        raise AttributeError("{} has no attribute '{}'.".format(
            type(self), name))

    def __setattr__(self, name, value):
        """override setting attributes for all elements in data"""
        try:
            # try to get data attribute
            # this call must use __getattribute__ to avoid infinite recursion
            data = self.__getattribute__('data')
            if name in data:
                data[name] = value
            elif name == "acc" and "acce" in data:
                data["acce"] = value
            elif name == "hmass" and "hrgm" in data:
                data["hrgm"] = value
            else:
                # normal behaviour if name not in data
                DirMixIn.__setattr__(self, name, value)
        except AttributeError:
            # if data not yet there, we want normal behaviour
            DirMixIn.__setattr__(self, name, value)

    def __dir__(self):
        """enable querying (and tab completion in ipython)
        for data dict entries"""
        seq = DirMixIn.__dir__(self)
        additional = ["acc", "hmass"]
        return list(seq) + additional + list(self.data.keys())

    def get_header_attribute( self, key, verbose=False ):
        # check header, then parameters, then config
        if key in self.header:
          return self.header[key]
        elif key in self.parameters:
          return self.parameters[key]
        elif key in self.config:
          return self.config[key]
        else:
          if verbose:
            print( "Could not find key '%s' in header, parameters, or config." % key )
          return None

    def load_header_hdf5( self, fileid, verbose, verboselevel=False ): 
        ff = h5py.File( self.files[fileid], 'r' )
        
        self.parameters = {}
        if '/Parameters' in ff:
          params = ff['/Parameters'].attrs
          for key in params:
            self.parameters[key] = params[key]

        self.config = {}
        if '/Config' in ff:
          params = ff['/Config'].attrs
          for key in params:
            self.config[key] = params[key]

        self.header = {}
        header = ff['/Header'].attrs
        for key in header:
          self.header[key] = header[key]

        self.nparticles    = self.get_header_attribute( 'NumPart_ThisFile', verbose=verbose )
        self.nparticlesall = np.uint64( self.get_header_attribute( 'NumPart_Total', verbose=verbose ) )
        self.npartTotalHighWord = self.get_header_attribute( 'NumPart_Total_HighWord', verbose=verbose )
        if self.npartTotalHighWord is not None:
          self.nparticlesall += np.left_shift(np.uint64(self.npartTotalHighWord), 32)

        self.masses = self.get_header_attribute( 'MassTable', verbose=verbose )

        self.time      = self.get_header_attribute( 'Time', verbose=verbose )
        self.redshift  = self.get_header_attribute( 'Redshift', verbose=verbose )
        self.boxsize   = self.get_header_attribute( 'BoxSize', verbose=verbose )
        self.num_files = self.get_header_attribute( 'NumFilesPerSnapshot', verbose=verbose )
        
        self.omega0      = self.get_header_attribute( 'Omega0', verbose=verbose )
        self.omegalambda = self.get_header_attribute( 'OmegaLambda', verbose=verbose )
        self.hubbleparam = self.get_header_attribute( 'HubbleParam', verbose=verbose )
        
        if 'Flag_Sfr' in header:
          self.flag_sfr = header['Flag_Sfr']
        if 'Flag_Cooling' in header:
          self.flag_cooling = header['Flag_Cooling']
        if 'Flag_StellarAge' in header:
          self.flag_stellarage = header['Flag_StellarAge']
        if 'Flag_Metals' in header:
          self.flag_metals = header['Flag_Metals']
        if 'Flag_Feedback' in header:
          self.flag_feedback = header['Flag_Feedback']
        if 'Flag_DoublePrecision' in header:
          self.flag_doubleprecision = header['Flag_DoublePrecision']
        
        self.UnitLength_in_cm         = self.get_header_attribute( 'UnitLength_in_cm', verbose=verbose )
        self.UnitMass_in_g            = self.get_header_attribute( 'UnitMass_in_g', verbose=verbose )
        self.UnitVelocity_in_cm_per_s = self.get_header_attribute( 'UnitVelocity_in_cm_per_s', verbose=verbose )
        
        if verbose and (self.UnitLength_in_cm is None or self.UnitMass_in_g is None or self.UnitVelocity_in_cm_per_s is None):
            print('couldnt find units in file')
        
        if self.boxsize > 0 and not self._center_initialized:
            self.set_center( np.ones(3) * 0.5 * self.boxsize )
            if self.twodim:
                self.center[2] = 0.0
            if self.onedim:
                self.center[1] = 0.0
                self.center[2] = 0.0
            self._center_initialized = True

        sort = self.nparticlesall.argsort()
        if self.nparticlesall[ sort[::-1] ][1] == 0:
            self.singleparticlespecies = True
        else:
            self.singleparticlespecies = False

        self.ntypes   = len( self.nparticlesall )
        self.loadtype = [value for value in self.loadtype if value in range(self.ntypes)]
        
        if not ((self.redshift > 0 and not self.cosmological == False) or (self.cosmological == True)):
            if self.convAddit:
                self.time /= self.hubbleparam
                self.boxsize /= self.hubbleparam

        if self.nparticlesall[1] > 0 and verbose and verboselevel:
            if self.singleparticlespecies:
                print( "Pure DARKNESS detected." )
            else:
                print( "DARKNESS detected." )

        if fileid == 0:
          if self.num_files != self.filecount:
            raise ValueError( "Number of files detected (%d) and num_files in the header (%d) are inconsistent." % (self.filecount, self.num_files) )

        ff.close()


        for ptype in range(self.ntypes):
            if not ptype in self.loadtype:
                self.nparticlesall[ptype] = 0
                self.nparticles[ptype] = 0

        self.npart = pylab.array( self.nparticles ).sum()
        self.npartall = pylab.array( self.nparticlesall ).sum()
        return

    def get_blocknames_hdf5(self, verbose=False):
        # get blocknames that are present
        self.blocknames = []
        self.vector_blocks = []
        # check all files to be sure to not miss fields
        for fileid in range(self.filecount):
            ff = h5py.File(self.files[fileid], 'r')
            nparticles = ff['/Header'].attrs['NumPart_ThisFile']
            for i in self.loadtype:
                if nparticles[i] > 0:
                    name = "PartType{:1d}".format(i)
                    if name not in ff:
                        continue
                    block = ff[name]
                    for key in block.keys():
                        if not key in gadget_snapshot.hdf5_name_conversion:
                            continue
                        ckey = gadget_snapshot.hdf5_name_conversion[key]
                        # continue if this block was already in another file
                        if ckey in self.blocknames:
                            continue
                        self.blocknames.append(ckey)
                        if ckey in ['xnuc', 'pass']:
                            if len(block[key].shape) > 1:
                                dim = block[key].shape[1]
                            else:
                                dim = 1
                            self.vector_blocks.append((ckey, dim))
                            if ckey == 'xnuc':
                                self.nspecies = dim
                            if ckey == 'pass':
                                self.npass = dim

    def load_data_chunk_hdf5( self, verbose, ptype, loadlist, fileid, loadonlyhalo ):
        if ptype < 0 or ptype > 6:
            print( "Error particle type cannot be %d" % ptype )
            return -1

        pfirst = 0
        plast = 0
        pcount = 0

        if loadonlyhalo >= 0:
            if self.halo_to_read[ptype] > 0:
                if self.halo_start_indizes[ptype] >= self.nparticles[ptype]:
                    self.halo_start_indizes[ptype] -= self.nparticles[ptype]
                else:                    
                    pfirst = self.halo_start_indizes[ptype]
                    plast  = pfirst + int(max( min( [self.nparticles[ptype] - pfirst, self.halo_to_read[ptype]] ), 0 ))
                    pcount = plast - pfirst
                    self.halo_start_indizes[ptype] = 0
                    self.halo_to_read[ptype] -= pcount
        else:
            pfirst = 0
            plast = int(self.nparticles[ptype])
            pcount = plast - pfirst
        
        if verbose:
            print( "Loading particles from %d to %d for file %d and type %d." % (pfirst,plast,fileid,ptype) )

        if pcount > 0:
                if self.filecount == 1 and not self.nommap:
                        #ff = h5py.File( self.files[fileid], 'r', driver='core' ) # doesn't work at the moment...
                        ff = h5py.File( self.files[fileid], 'r' )
                else:
                        ff = h5py.File( self.files[fileid], 'r' )

                pname = "PartType%d" % ptype
                block = ff[pname]

                if verbose:
                        print( "Doing file %d." % (fileid) )

                for key in block.keys():
                       if not key in gadget_snapshot.hdf5_name_conversion:
                              if fileid == 0 and verbose:
                                     print( "Key %s not found in conversion table, skipping." % key )
                              continue

                       ckey = gadget_snapshot.hdf5_name_conversion[ key ]
                       partkey = gadget_snapshot.hdf5_part_field[ ckey ]
                       
                       if verbose:
                           print( "Loading %s (%s) for particle type %d." % (ckey, key, ptype) )

                       if not loadlist or ckey in loadlist:
                                if key in ['ParticleIDs']:
                                    datatype = 'uint64'
                                    dset = block[key]
                                    print('\n\n\t>>DEBUGGING<<\n')
                                    print(type(dset))
                                    # with dset.astype('uint64'):
                                    #     data = dset[:]
                                    data = dset.astype('uint64')[:] #SRW edit 21/11/24
                                else:
                                    datatype = block[key].dtype
                                    if self.forcesingleprec and self.flag_doubleprecision:
                                        if datatype == 'float64':
                                            datatype = 'float32'
                                    if self.forcedoubleprec and not self.flag_doubleprecision:
                                        if datatype == 'float32':
                                            datatype = 'float64'
                                    if self.convert_posvel_to_double:
                                        if ckey == 'pos' or ckey == 'vel':
                                            datatype='float64'
                                    data = block[key][:]
                                
                                if ckey in self.data:
                                    self.data[ ckey ][self.offset[partkey]:self.offset[partkey]+pcount] = data[pfirst:plast]
                                else:
                                    shape = np.array(data.shape).copy()
                                    shape[0] = self.npartalloc[partkey]

                                    self.data[ ckey ] = np.empty( shape, dtype=datatype )
                                    self.data[ ckey ][self.offset[partkey]:self.offset[partkey]+pcount] = data[pfirst:plast]

                                if verbose:
                                    print( "Loaded %s (%s) for particle type %d. Datatype %s, size %d bytes, current off %d," % (ckey, key, ptype, datatype, self.data[ckey].nbytes, self.offset[partkey] + self.nparticles[ptype]), 'shape', self.data[ckey].shape )


                # fix for masses...
                ckey = gadget_snapshot.hdf5_name_conversion[ 'Masses' ]
                partkey = gadget_snapshot.hdf5_part_field[ ckey ]
                if not block.get( 'Masses' ) and self.masses[ptype] > 0 and ( ckey in loadlist or not loadlist ):
                       if ckey in self.data:
                                self.data[ ckey ][self.offset[partkey]:self.offset[partkey]+pcount] = self.masses[ptype]
                       else:
                                datatype = self.masses.dtype
                                if self.forcesingleprec and self.flag_doubleprecision:
                                        if datatype == 'float64':
                                                datatype = 'float32'

                                self.data[ ckey ] = np.empty( self.npartalloc[partkey], dtype=datatype )
                                self.data[ ckey ][self.offset[partkey]:self.offset[partkey]+pcount] = self.masses[ptype]

                       if verbose:
                                print( 'Added mass block for particle type %d with constant mass %g.' % (ptype, self.masses[ptype]) )

                # fix for type...
                if 'type' in self.data:
                        self.data['type'][self.offset['all']:self.offset['all']+pcount] = ptype
                else:
                        self.data['type'] = np.empty( self.npartalloc['all'], dtype='int8' )
                        self.data['type'][self.offset['all']:self.offset['all']+pcount] = ptype

                if verbose:
                        print( 'Added type block for particles of type %d total size' % (ptype), self.data['type'].shape )

                if ptype == 0:
                        self.offset['gas'] += pcount
                        self.offset['baryons'] += pcount
                if ptype in [1, 2, 3]:
                        self.offset['dm'] += pcount
                if ptype == 4:
                        self.offset['stars'] += pcount
                        self.offset['baryons'] += pcount
                if ptype == 5:
                        self.offset['bh'] += pcount
                if ptype == 6:
                    self.offset['tracers'] += pcount

                self.offset['all'] += pcount

                ff.close()

        return

    def _lazy_load_data_hdf5(self, key):
        """wrapper function for lazy loading of hdf5 snaps

        needed as direct method of the snapshot object to be able to get a
        weak reference
        """
        return self.load_data_hdf5(
                verbose=self.verbose, loadlist=[key], chunknum=self.chunk,
                loadonlyhalo=self.loadonlyhalo)

    def load_data_hdf5( self, verbose, loadlist=[], chunknum=None, loadonlyhalo=-1 ):
        if loadonlyhalo >= 0 and self.offsets is None:
            self.get_offsets(verbose=verbose)
        
        if self.lazy_load:
            if len(loadlist) > 1:
                raise ValueError("loadlist should contain only one element"
                                 " for lazy_load!")
            # check for volume
            if loadlist[0] == 'vol':
                if "rho" in self.data:
                    return self.mass[:self.nparticlesall[0]] / self.data["rho"].astype('f8')
                else:
                    return None
            # if type is wanted, get masses
            loadtype = False
            if loadlist[0] == 'type':
                loadlist[0] = 'mass'
                loadtype = True
            # store data array
            _data = self.data
        self.data = {}

        if self.ntypes == 7:
            self.offset = {'all' : 0, 'gas' : 0, 'dm' : 0, 'stars' : 0, 'baryons' : 0, 'bh' : 0, 'tracers' : 0 }
        else:
            self.offset = {'all' : 0, 'gas' : 0, 'dm' : 0, 'stars' : 0, 'baryons' : 0, 'bh' : 0 }

        if chunknum is None:
            firstFile = np.zeros( len(self.loadtype), dtype='int32' )
            
            if loadonlyhalo >= 0:
                self.halo_start_indizes = self.subfind.data['flty'][:loadonlyhalo,:].sum(axis=0)
                self.halo_to_read = self.subfind.data['flty'][loadonlyhalo,:].copy()
                
                for ptype in range(self.ntypes):
                    if not ptype in self.loadtype:
                        self.halo_start_indizes[ptype] = 0
                        self.halo_to_read[ptype] = 0
                
                self.npartalloc = {'all' : self.halo_to_read.sum(),
                    'gas' : self.halo_to_read[0],
                    'dm' : self.halo_to_read[[value for value in [1,2,3] if value in range(self.ntypes)]].sum(),
                    'stars' : self.halo_to_read[[value for value in [4] if value in range(self.ntypes)]],
                    'baryons' : self.halo_to_read[[value for value in [0,4] if value in range(self.ntypes)]].sum(),
                    'bh' : self.halo_to_read[[value for value in [5] if value in range(self.ntypes)]]}
                if self.ntypes > 6:
                    self.npartalloc[ 'tracers' ] = self.halo_to_read[6]
                else:
                    self.npartalloc[ 'tracers' ] = 0
                
                if verbose:
                    print( "I will load the following number of particles:", self.halo_to_read )
                
                if self.offsets is not None:
                    for itype in range(len(self.loadtype)):
                        ptype = self.loadtype[itype]
                        
                        while firstFile[itype] < self.filecount-1 and self.offsets.files[firstFile[itype]][ptype] < self.halo_start_indizes[ptype]:
                            firstFile[itype] += 1
                        
                        # offsets[ "files" ]     = f['FileOffsets']['SnapByType'][:]
                        # offsets[ "groups" ]    = f['Group']['SnapByType'][:]
                        # offsets[ "subhalos" ]  = f['Subhalo']['SnapByType'][:]
                    
                    if verbose:
                        print( "I will start from files:", firstFile )
            else:

                self.npartalloc = {'all' : self.nparticlesall[self.loadtype].sum(),
                    'gas' : self.nparticlesall[0],
                    'dm' : self.nparticlesall[[value for value in [1,2,3] if value in range(self.ntypes)]].sum(),
                    'stars' : self.nparticlesall[[value for value in [4] if value in range(self.ntypes)]],
                    'baryons' : self.nparticlesall[[value for value in [0,4] if value in range(self.ntypes)]].sum(),
                    'bh' : self.nparticlesall[[value for value in [5] if value in range(self.ntypes)]]}
                if self.ntypes > 6:
                    self.npartalloc[ 'tracers' ] = self.nparticlesall[6]
                else:
                    self.npartalloc[ 'tracers' ] = 0

            for ptype in self.loadtype:
                if self.nparticlesall[ptype] > 0:
                    for fileid in range( self.filecount ):
                        self.load_header_hdf5( fileid, verbose )
                        self.load_data_chunk_hdf5( verbose, ptype, loadlist, fileid, loadonlyhalo )
                        
                        # check if we are already done loading this particle type
                        if loadonlyhalo >= 0:
                            if self.halo_to_read[ptype] == 0:
                                break
        else:
            if chunknum < 0 or chunknum >= self.filecount:
                raise Exception( 'Invalid chunknum=%d. It must be greater than 0 or less than %d' % (chunknum, self.filecount) )

            for ptype in self.loadtype:
                if self.nparticlesall[ptype] > 0:
                    self.load_header_hdf5( chunknum, verbose )

                    self.npartalloc = {'all' : self.nparticles[self.loadtype].sum(),
                        'gas' : self.nparticles[0],
                        'dm' : self.nparticles[[1,2,3]].sum(),
                        'stars' : self.nparticles[4],
                        'baryons' : self.nparticles[[0,4]].sum(),
                        'bh' : self.nparticles[5]}
                    if self.ntypes > 6:
                        self.npartalloc[ 'tracers' ] = self.nparticlesall[6]
                    else:
                        self.npartalloc[ 'tracers' ] = 0

                    self.load_data_chunk_hdf5( verbose, ptype, loadlist, chunknum, loadonlyhalo )
            
            self.nparticlesall = self.nparticles
            self.npartall      = self.npart

        if loadonlyhalo >= 0:
            self.nparticles = self.subfind.data['flty'][loadonlyhalo,:].copy()
            self.nparticlesall = self.subfind.data['flty'][loadonlyhalo,:].copy()
        
        for ptype in range(self.ntypes):
            if not ptype in self.loadtype:
                self.nparticlesall[ptype] = 0
                self.nparticles[ptype] = 0
        
        self.npart = self.nparticles.sum()
        self.npartall = self.nparticlesall.sum()

        if verbose:
            print( 'snapshot file read' )

        if self.lazy_load:
            result = self.data[loadlist[0]]
            if loadlist[0] in self.transformation_factors:
                result *= self.transformation_factors[loadlist[0]]
            # want to load type
            if loadtype:
                result = self.data['type']
            _data['type'] = self.data['type']
            # store back the data dict
            self.data = _data
            # return loaded data to calling LazyDict
            return result
        return

    def get_blocks( self, fileid, verbose=False ):
        swap, endian = endianness_check( self.files[fileid] )
        
        self.blocks = {}
        self.origdata = []
        f = open( self.files[fileid], 'rb' )
        
        fpos = np.int64( f.tell() )
        s = f.read(16)
        while len(s) > 0:
            fheader, name, length, ffooter = struct.unpack( endian + "i4sii", s )
            if sys.version_info.major == 3:
                name = name.decode()
            self.blocks[ name ] = fpos
            
            if length < 0:
                length = np.int64( length ) + 0x100000000

            if verbose:
                print("Block %s, fpos=%d, length=%d." % (name, fpos, length))

            length = self.get_block_length( fileid, fpos + 16, name, length, endian, verbose=verbose )
            if length < 0:
                return False

            if verbose:
                print("Block %s, length %d, offset %d." %(name, length, fpos))

            f.seek( length, 1 )
            fpos = np.int64( f.tell() )
            s = f.read(16)
            
            self.origdata += [name]

        if verbose:
            print("%d blocks detected." % len(list(self.blocks)))
        return True

    def get_block_names_type2(self, verbose=False):
        self.blocknames = []
        self.vector_blocks = []

        # check all files
        for fileid in range(self.filecount):
            swap, endian = endianness_check( self.files[fileid] )
            self.load_header(fileid, verbose)

            f = open( self.files[fileid], 'rb' )

            fpos = np.int64( f.tell() )
            s = f.read(16)
            while len(s) > 0:
                fheader, name, length, ffooter = struct.unpack( endian + "i4sii", s )
                if sys.version_info.major == 3:
                    name = name.decode()
                length = self.get_block_length( fileid, fpos + 16, name, length, endian, verbose=verbose )

                f.seek( length, 1 )
                fpos = np.int64( f.tell() )
                s = f.read(16)

                if name != 'HEAD':
                    blockname = name.strip().lower()
                    if blockname in self.blocknames:
                        continue
                    self.blocknames += [blockname]

                if name in ['XNUC', 'PASS']:
                    # cmopute dimenstions for vector blocks
                    if self.flag_doubleprecision:
                        elementsize = 8
                    else:
                        elementsize = 4
                    npart, npartall = self.get_block_size_from_table( name )
                    elements_tot = (length - 8) // elementsize
                    dim = elements_tot // npart
                    self.vector_blocks.append((name.strip().lower(), dim))
                    if name == 'XNUC':
                        self.nspecies = dim
                    if name == 'PASS':
                        self.npass = dim

    def get_block_length( self, fileid, start, name, length, endian, verbose=False ):
        if verbose:
            print("Getting block length of block %s." % (name))
            
        if self.check_block_length( fileid, start, length, endian, verbose ):
            return length

        if verbose:
            print("First check failed.")

        if name == "XNUC" or name == "PASS":
            bs, bsall = self.get_block_size_from_table( name )
            if self.flag_doubleprecision:
                ele = 8
            else:
                ele = 4

            for dim in range( 500 ):
                length = np.int64( bs * dim * ele + 8 )
                if self.check_block_length( fileid, start, length, endian, verbose ):
                    if verbose:
                        print("Block %s solved, bs=%d, dim=%d." % (name, bs, dim))
                    return length

            print("Error determining the length of block %s." % (name))
            return -1
            
        else:
            bs, bsall = self.get_block_size_from_table( name )
            dim = self.get_block_dim_from_table( name )
            if self.flag_doubleprecision:
                ele = 8
            else:
                ele = 4

            length = bs * dim * ele + 8
            
            if self.check_block_length( fileid, start, length, endian, verbose ):
                return length
            else:
                print("Error determining the length of block %s." % (name))
                return -1
        
        return length

    def check_block_length( self, fileid, start, length, endian, verbose=False ):
        if verbose:
            print("Checking block start=%d, length=%d, endian=%s." % (start, length, endian))
        
        f = open( self.files[fileid], 'rb' )

        f.seek( start, 0 )
        fheader, = struct.unpack( endian + "i", f.read(4) )
        f.seek( length-8, 1 )
        ffooter, = struct.unpack( endian + "i", f.read(4) )
        if fheader != ffooter:
            if verbose:
                print("Header=%d, Footer=%d." % (fheader, ffooter))
            f.close()
            return False

        s = f.read(4)
        if len( s ) == 4:
            fheader, = struct.unpack( endian + "i", s )
            if fheader != 8:
                f.close()
                return False

        f.close()
        return True

    def load_header( self, fileid, verbose=False ):
        self.ntypes = 6
        
        swap, endian = endianness_check( self.files[fileid] )
        
        f = open( self.files[fileid], 'rb' )
        s = f.read(16)
        while len(s) > 0:
            fheader, name, length, ffooter = struct.unpack( endian + "i4sii", s )

            if name != b"HEAD":
                f.seek( length, 1 )
                s = f.read( 16 )
            else:
                f.seek( 4, 1 ) # skip fortran header of data block
                s = f.read(24)
                self.nparticles = pylab.array( struct.unpack( endian + "6i", s ) )
                s = f.read(48)
                self.masses = pylab.array( struct.unpack( endian + "6d", s ) )
                s = f.read(24)
                self.time, self.redshift, self.flag_sfr, self.flag_feedback = struct.unpack( endian + "ddii", s )
                s = f.read(24)
                self.nparticlesall = pylab.array( struct.unpack( endian + "6i", s ) )
                s = f.read(16)
                self.flag_cooling, num_files, self.boxsize = struct.unpack( endian + "iid", s )
                s = f.read(24)
                self.omega0, self.omegalambda, self.hubbleparam = struct.unpack( endian + "ddd", s )
                s = f.read(8)
                self.flag_stellarage, self.flag_metals = struct.unpack( endian + "ii", s )
                s = f.read(24)
                self.nparticlesallhighword = pylab.array( struct.unpack( endian + "6i", s ) )
                s = f.read(12)
                self.flag_entropy_instead_u, self.flag_doubleprecision, self.flag_lpt_ics = struct.unpack( "iii", s )
                s = f.read(52)

                if self.boxsize > 0 and not self._center_initialized:
                    self.set_center( [0.5 * self.boxsize, 0.5 * self.boxsize, 0.5 * self.boxsize] )
                    if self.twodim:
                        self.center[2] = 0.0
                    if self.onedim:
                        self.center[1] = 0.0
                        self.center[2] = 0.0
                    self._center_initialized = True

                self.nparticles = pylab.array( self.nparticles )
                self.nparticlesall = pylab.array( self.nparticlesall )

                if self.nparticlesall.sum() == 0:
                    self.nparticlesall = self.nparticles

                sort = self.nparticlesall.argsort()
                if self.nparticlesall[ sort[::-1] ][1] == 0:
                    self.singleparticlespecies = True
                else:
                    self.singleparticlespecies = False

                if self.nparticlesall[1] > 0 and verbose:
                    if self.singleparticlespecies:
                        print("Pure DARKNESS detected.")
                    else:
                        print("DARKNESS detected.")

                f.close()
                s = "" # stop reading

                self.npart = pylab.array( self.nparticles ).sum()
                self.npartall = pylab.array( self.nparticlesall ).sum()

                if verbose:
                    print("nparticlesall:", self.nparticlesall, "sum:", self.npartall)
                    print("masses:", self.masses)

                if fileid == 0:
                    if (num_files != self.filecount) and not (num_files == 0 and self.filecount == 1):
                        raise Exception( "Number of files detected (%d) and num_files in the header (%d) are inconsistent." % (self.filecount, num_files) )
        if verbose:
            print("Snapshot contains %d particles." % self.npartall)
        return

    def load_data_mmap( self, verbose=False ):
        # memory mapping works only if there is only one file
        swap, endian = endianness_check( self.files[0] )
        self.data = {}

        self.get_blocks( 0, verbose=verbose )
        f = open( self.files[0], 'rb' )
        for block in self.blocks:
            # skip some blocks
            if block in self.datablocks_skip:
                continue

            if verbose:
                print("Starting to work on block %s." % (block))

            f.seek( self.blocks[block], 0 )
            fheader, name, length, ffooter = struct.unpack( endian + "i4sii", f.read(16) )
            if sys.version_info.major == 3:
                name = name.decode()
            if length < 0:
                length = np.int64( length ) + 0x100000000
            length = self.get_block_length( 0, np.int64( f.tell() ), name, length, endian )

            if block in self.datablocks_int32:
                blocktype = "i4"
                elementsize = 4
            else:
                if self.flag_doubleprecision:
                    blocktype = 'f8'
                    elementsize = 8
                else:
                    blocktype = 'f4'
                    elementsize = 4

            nsum = 0
            for ptype in range( self.ntypes ):
                if self.nparticles[ptype] == 0:
                    continue
                    
                if not self.parttype_has_block( ptype, name ):
                    continue

                nsum += self.nparticlesall[ptype]
                
            elements = (length-8)/elementsize
            dim = int(elements / nsum)
            
            if verbose:
                print("Loading block %s, offset %d, length %d, elements %d, dimension %d, particles %d/%d." % (block, self.blocks[block], length, elements, dim, nsum, self.nparticlesall.sum()))

            offset = np.int64( f.tell() ) + 4
            blockname = block.strip().lower()
            if dim == 1:
                self.data[ blockname ] = np.memmap( self.files[0], offset=offset, mode='c', dtype=endian+blocktype, shape=(nsum) )
            else:
                self.data[ blockname ] = np.memmap( self.files[0], offset=offset, mode='c', dtype=endian+blocktype, shape=(nsum, dim) )
        return

    def _lazy_load_data(self, key):
        """wrapper function for lazy loading

        needed as direct method of the snapshot object to be able to get a
        weak reference
        """
        return self.load_data(
                verbose=self.verbose, loadlist=["{:<4}".format(key.upper())])

    def load_data( self, verbose=False, nommap=False, loadlist=[] ):
        swap, endian = endianness_check( self.files[0] )
        if self.lazy_load:
            if len(loadlist) > 1:
                raise ValueError("loadlist should contain only one element"
                                 " for lazy_load!")
            # check for volume
            if loadlist[0] == 'VOL ':
                return self.mass[:self.nparticlesall[0]] / self.rho.astype('f8')
            # store data array
            _data = self.data
            blockname = ""
        self.data = {}

        nparttot = pylab.zeros( self.ntypes, dtype='int32' )  
        for fileid in range( self.filecount ):
            self.load_header( fileid, verbose=verbose )
            self.get_blocks( fileid, verbose=verbose )
            
            f = open( self.files[fileid], 'rb' )
            for block in self.blocks:
                # skip some blocks
                if block in self.datablocks_skip:
                    continue

                if loadlist and not block in loadlist:
                    continue

                if verbose:
                    print("Loading block %s of file %s." % (block, fileid))

                f.seek( self.blocks[block], 0 )
                fheader, name, length, ffooter = struct.unpack( endian + "i4sii", f.read(16) )
                if sys.version_info.major == 3:
                    name = name.decode()
                length = self.get_block_length( fileid, np.int64( f.tell() ), name, length, endian )
                f.seek( 4, 1 ) # skip fortran header of data field

                if block in self.datablocks_int32:
                    blocktype = "i4"
                    elementsize = 4
                else:
                    if self.flag_doubleprecision:
                        blocktype = 'f8'
                        elementsize = 8
                    else:
                        blocktype = 'f4'
                        elementsize = 4
                
                nsum = 0
                for ptype in range( self.ntypes ):
                    if self.nparticles[ptype] == 0:
                        continue
                    
                    if not self.parttype_has_block( ptype, name ):
                        continue
                    
                    npart, npartall = self.get_block_size_from_table( name )
                    # old: from table -> problem with xnuc and pass
                    # dim = self.get_block_dim_from_table( name )
                    npartptype = self.nparticles[ptype]
                    # total number of elements
                    elements_tot = (length - 8) // elementsize
                    dim = elements_tot // npart
                    # elements of current particle type
                    elements = npartptype * dim
                    
                    if verbose:
                        print("Loading block %s, offset %d, length %d, elements %d, dimension %d, type %d, particles %d/%d." % (block, self.blocks[block], length, elements, dim, ptype, npartptype, npartall))
                    
                    blockname = block.strip().lower()
                    if blockname not in self.data:
                        if dim == 1:
                            self.data[ blockname ] = pylab.zeros( npartall, dtype=blocktype )
                        else:
                            self.data[ blockname ] = pylab.zeros( (npartall, dim), dtype=blocktype )

                    if blocktype == 'f4':
                        self.data[ blockname ] = self.data[ blockname ].astype( 'float64' ) # change array type to float64

                    lb = nsum + nparttot[ptype]
                    ub = lb + npartptype
                    
                    if verbose:
                        print("Block contains %d elements (length=%g, elementsize=%g, lb=%d, ub=%d)." % (elements,length,elementsize,lb,ub))

                    if dim == 1:
                        self.data[ blockname ][lb:ub] = np.fromfile( f, dtype=endian+blocktype, count=elements )
                    else:
                        self.data[ blockname ][lb:ub,:] = np.fromfile( f, dtype=endian+blocktype, count=elements ).reshape( npartptype, dim )

                    nsum += self.nparticlesall[ptype]

            nparttot += self.nparticles

        if self.filecount > 1:
            self.npart = self.npartall
        print('%d particles loaded.' % self.npartall)
        if self.lazy_load:
            result = self.data[blockname]
            if loadlist[0] in self.transformation_factors:
                result *= self.transformation_factors[loadlist[0].strip().lower()]
            # store back the data dict
            self.data = _data
            # return loaded data to calling LazyDict
            return result
        return

    def get_block_size_from_table( self, block ):
        npart = 0
        npartall = 0
        if block in ["POS ", "VEL ", "ID  ", "POT ", "TSTP"]:
            # present for all particles
            npart = self.nparticles.sum()
            npartall = self.nparticlesall.sum()
        if block == "MASS":
            i, = np.where( self.masses == 0. )
            npart = self.nparticles[i].sum()
            npartall = self.nparticlesall[i].sum()
        if block in ["U   ","RHO ", "NE  ", "NH  ", "HSML", "SFR ", "Z   ", "XNUC", "PRES", "VORT", "VOL ", "HRGM", "REF ", "DIVV", "ROTV", "DUDT", "BFLD", "DIVB", "PSI ", "VDED", "REF ", "PASS", "TEMP", "GRAR", "GRAP", "GRAV", "GRAB", "NONN", "TNGB", "ABVC", "SIDX", "CMCE", "SPIN", "LPOS", "CREN", "CHI ", "AFLD", "ASHX", "ASHY", "ASHZ", "ACCE", "PMAC", "TRAC"]:
            # present for hydro particles
            npart += self.nparticles[0]
            npartall += self.nparticlesall[0]
        if block in ["VODE"]:
            npart += self.nparticles[1]
            npartall += self.nparticlesall[1]
        if block in ["AGE ", "Z   ", "BPOS", "BVEL", "BDENS"]:
            # present for star particles
            npart += self.nparticles[4]
            npartall += self.nparticlesall[4]
        if block in ["BHMA", "BHMD"]:
            #present for black hole particles
            npart += self.nparticles[5]
            npartall += self.nparticlesall[5]
        if block in ["TRID", "PRID", "FLDQ"]:
            npart += self.nparticles[6]
            npartall += self.nparticlesall[6]

        if self.tracer:
            if block in ["MASS"]:
                npart -= self.nparticles[ self.tracer ]
                npartall -= self.nparticlesall[ self.tracer ]
        return npart, npartall
    
    def parttype_has_block( self, ptype, block ):
        if block in ["POS ", "VEL ", "ID  ", "MASS", "POT ", "TSTP"]:
            if self.tracer and block in ["MASS"]:
                if ptype == self.tracer:
                    return False
            if block == "MASS" and self.masses[ptype] > 0.:
                return False
            return True
        elif block in ["U   ","RHO ", "NE  ", "NH  ", "HSML", "SFR ", "Z   ", "XNUC", "PRES", "VORT", "VOL ", "HRGM", "REF ", "DIVV", "ROTV", "DUDT", "BFLD", "DIVB", "PSI ", "VDED", "PASS", "TEMP", "ACCE", "PMAC", "TRAC", "P   ", "GRAR", "GRAP", "GRAV", "GRAB", "NONN", "TNGB", "ABVC", "SIDX", "CMCE", "SPIN", "LPOS", "CREN", "CHI ", "AFLD", "ASHX", "ASHY", "ASHZ"]:
            if ptype == 0:
                return True
            else:
                return False
        elif block in ["VODE"]:
            if ptype == 1:
                return True
            else:
                return False
        elif block in ["AGE ", "Z   ", "BPOS", "BVEL", "BDENS"]:
            if ptype == 4:
                return True
            else:
                return False
        elif block in ["BHMA", "BHMD"]:
            if ptype == 5:
                return True
            else:
                return False
        elif block in ["TRID", "PRID", "FLDQ"]:
            if ptype == 6:
                return True
            else:
                return False
        else:
                return False
        
    def get_block_dim_from_table( self, block ):
        if block in ["POS ", "VEL ", "DIVV", "VORT", "BFLD", "GRAR", "GRAP", "CMCE", "SPIN", "LPOS", "BPOS", "BVEL", "AFLD", "ASHX", "ASHY", "ASHZ", "ACCE", "PMAC", "TRAC"]:
            return 3
        elif block in ["GRAV","GRAB"]:
            return 9
        else:
            return 1

    def convenience( self ):
        if not self.lazy_load:
            if not "vol" in self.data and "rho" in self.data and "mass" in self.data:
                self.data["vol"] = self.mass[:self.nparticlesall[0]] / self.rho.astype('f8')

            if "xnuc" in self.data:
                if self.data['xnuc'].ndim > 1:
                    self.nspecies = pylab.shape( self.data["xnuc"] )[1]
                    for i in range( self.nspecies ):
                        name = "xnuc%02d" % (i)
                        self.data[name] = self.data["xnuc"][:,i]
                else:
                    self.nspecies = 1

            if "pass" in self.data:
                if self.data['pass'].ndim > 1:
                    self.npass = pylab.shape( self.data["pass"] )[1]
                    for i in range( self.npass ):
                        name = "pass%02d" % (i)
                        self.data[name] = self.data["pass"][:,i]
                else:
                    self.npass = 1

        if not self.hdf5:
            self.data['type'] = pylab.zeros( self.npartall, dtype="int32" )
            for ptype in range( self.ntypes ):
                start = self.nparticlesall[:ptype].sum()
                end   = self.nparticlesall[ptype] + start
                self.data['type'][start:end] = ptype
        return
    
    def cosmology_init( self ):
        self.cosmo = CosmologicalFactors( my_h = self.hubbleparam, my_OmegaMatter = self.omega0, my_OmegaLambda = self.omegalambda )
        self.cosmo.SetLookbackTimeTable()
        return
    
    def cosmology_get_lookback_time_from_a( self, a, is_flat=False, quicklist=False ):
        if quicklist:
            self.cosmo = CosmologicalFactors( my_h = self.hubbleparam, my_OmegaMatter = self.omega0, my_OmegaLambda = self.omegalambda )
            self.cosmo.SetLookbackTimeTable()            
            return self.cosmo.LookbackTime_a_in_Gyr( a, is_flat )
        else:
            return self.cosmo.LookbackTime_a_in_Gyr( a, is_flat )

    def set_transformation_factors_non_comoving( self ):
        if self.hubbleparam <= 0:
            return
        self.transformation_factors['pos'] = 1.0 / self.hubbleparam
        self.transformation_factors['acce'] = self.hubbleparam
        self.transformation_factors['rho'] = self.hubbleparam**2
        self.transformation_factors['vol'] = 1.0 / self.hubbleparam**3
        self.transformation_factors['pres'] = self.hubbleparam**2
        self.transformation_factors['mass'] = 1.0 / self.hubbleparam
        self.masses /= self.hubbleparam
        self.center /= self.hubbleparam
        self.transformation_factors['hrgm'] = 1.0 / self.hubbleparam
        self.transformation_factors['gima'] = 1.0 / self.hubbleparam
        self.transformation_factors['bfld'] = self.hubbleparam
        self.transformation_factors['divb'] = self.hubbleparam**2
        self.transformation_factors['dvba'] = self.hubbleparam**2
        self.transformation_factors['bcmq'] = 1.0 / self.hubbleparam
        self.transformation_factors['bcmr'] = 1.0 / self.hubbleparam
        self.transformation_factors['bhhs'] = 1.0 / self.hubbleparam
        self.transformation_factors['bhpr'] = self.hubbleparam**2
        self.transformation_factors['bhro'] = self.hubbleparam**2
        self.transformation_factors['bhma'] = 1.0 / self.hubbleparam
        self.transformation_factors['bhmz'] = 1.0 / self.hubbleparam
        self.transformation_factors['bfgr'] = self.hubbleparam**2
        self.transformation_factors['crpg'] = self.hubbleparam**3
        self.transformation_factors['cmce'] = 1.0 / self.hubbleparam
        self.transformation_factors['grar'] = self.hubbleparam**3
        self.transformation_factors['grap'] = self.hubbleparam**3
        self.transformation_factors['tstp'] = 1.0 / self.hubbleparam
        self.transformation_factors['divv'] = self.hubbleparam
        self.transformation_factors['grav'] = self.hubbleparam
        self.transformation_factors['vort'] = self.hubbleparam
        return

    def set_transformation_factors_comoving( self ):
        self.transformation_factors['pos']  = self.time / self.hubbleparam
        self.transformation_factors['cmce'] = self.time / self.hubbleparam
        self.transformation_factors['vel']  = sqrt( self.time )
        self.transformation_factors['acce'] = self.hubbleparam
        # for lazy load: loading age always
        if 'age' in self.data:
            self.transformation_factors['bpos'] = self.data['age'][:,None] / self.hubbleparam
            self.transformation_factors['bvel'] = 1.0 / self.data['age'][:,None]
            self.transformation_factors['bdens'] = 1.0 / (self.data['age']**3 / self.hubbleparam**2)
        self.transformation_factors['vode'] = 1.0 / (self.time**3 / self.hubbleparam**2)
        self.transformation_factors['rho'] = 1.0 / (self.time**3 / self.hubbleparam**2)
        self.transformation_factors['vol'] = self.time**3 / self.hubbleparam**3
        self.transformation_factors['pres'] = 1.0 / (self.time**3 / self.hubbleparam**2)
        self.transformation_factors['mass'] = 1.0 / self.hubbleparam
        self.masses /= self.hubbleparam
        self.center *= self.time / self.hubbleparam
        self.transformation_factors['bfld'] = 1.0 / (self.time**2 / self.hubbleparam)
        self.transformation_factors['curb'] = 1.0 / (self.time**2 / self.hubbleparam)
        self.transformation_factors['divb'] = 1.0 / (self.time**3 / self.hubbleparam**2)
        self.transformation_factors['dvba'] = 1.0 / (self.time**3 / self.hubbleparam**2)
        self.transformation_factors['hrgm'] = 1.0 / self.hubbleparam
        self.transformation_factors['gima'] = 1.0 / self.hubbleparam
        self.transformation_factors['bcmq'] = 1.0 / self.hubbleparam
        self.transformation_factors['bcmr'] = 1.0 / self.hubbleparam
        self.transformation_factors['bhhs'] = self.time / self.hubbleparam
        self.transformation_factors['bhpr'] = 1.0 / (self.time**3 / self.hubbleparam**2)
        self.transformation_factors['bhro'] = 1.0 / (self.time**3 / self.hubbleparam**2)
        self.transformation_factors['bhma'] = 1.0 / self.hubbleparam
        self.transformation_factors['bhmz'] = 1.0 / self.hubbleparam
        self.transformation_factors['gimr'] = 1.0 / self.hubbleparam
        self.transformation_factors['gimz'] = 1.0 / self.hubbleparam
        self.transformation_factors['gmmz'] = 1.0 / self.hubbleparam
        return

    def apply_transformation_factors(self):
        for k, factor in self.transformation_factors.items():
            if k in self.data:
                self.data[k] *= np.double(factor)

    def calc_mean_a( self, speciesfile="../species.txt" ):
        sp = loaders.load_species( speciesfile )

        if sp['count'] != self.nspecies:
            print("Number of species in speciesfile (%d) and snapshot (%d) don't match." % (sp['count'],self.nspecies))

        self.data['mean_a'] = np.zeros( self.nparticlesall[0] )
        for i in range( self.nspecies ):
            self.data['mean_a'] += self.data['xnuc'][:,i] * sp['na'][i]
        return

    """
    def find_files_to_load_from_offsets( self ):
      chunks_to_load = set()
      for ptype in s.loadtype:
        snap_part_start = self.offsets_particles[s.chunk, ptype]
        
        if s.chunk == s.num_files-1:
          snap_part_end = snap_part_start + s.nparticlesall[ptype]
        else:
          snap_part_end = self.offsets_particles[s.chunk+1, ptype]
        
        for ifile in range( s.num_files ):
          if self.offsets_halo[ifile] == self.offsets_halo[ifile+1]:
            continue
          
          group_part_start = self.offsets_halos_particles[self.offsets_halo[ifile], ptype]
          
          if group_part_start >= snap_part_end:
            #print( "DEBUG: A:", ptype, ifile, snap_part_start, snap_part_end, group_part_start, group_part_end )
            break
          
          if ifile == s.num_files-1:
            #print( "DEBUG: B:", ptype, ifile )
            chunks_to_load.add( ifile )
            break
          
          group_part_end = self.offsets_halos_particles[self.offsets_halo[ifile+1], ptype]
          
          if group_part_end <= snap_part_start:
            #print( "DEBUG: C:", ptype, ifile, snap_part_start, snap_part_end, group_part_start, group_part_end )
            continue
          
          #print( "DEBUG: D:", ptype, ifile, snap_part_start, snap_part_end, group_part_start, group_part_end )
          chunks_to_load.add( ifile )
          
          #print( ifile, self.offsets_halos_particles[self.offsets_halo[ifile+0],0], self.offsets_halos_particles[self.offsets_halo[ifile+1],0], self.offsets_particles[self.snapchunk,0], self.offsets_particles[self.snapchunk+1,0] )
      
      list_of_files = list( chunks_to_load )
      list_of_files.sort()
      return list_of_files
    """
    
    def get_offsets( self, verbose=False ):
        directory = path.dirname(self.filename)
        snapnr    = int( directory[-3:] )
        
        fname = "%s/../../postprocessing/offsets/offsets_%03d.hdf5" % (directory, snapnr)
        if not path.exists( fname ):
            fname = "%s/../postprocessing/offsets/offsets_%03d.hdf5" % (directory, snapnr)
            if not path.exists( fname ):
                fname = None
        
        offsets = {}
        
        if fname:
            if verbose:
                print( "VERBOSE: Loading offsets from file '%s'." % fname )
        
            with h5py.File( fname, 'r' ) as f:
                offsets[ "files" ]     = f['FileOffsets']['SnapByType'][:]
                offsets[ "groups" ]    = f['Group']['SnapByType'][:]
                offsets[ "subhalos" ]  = f['Subhalo']['SnapByType'][:]
                
            self.offsets = dict2obj( offsets )
            
            if verbose:
                print( "VERBOSE: Loaded offsets." )
        else:
            print( "ERROR: Creating offsets files on the file is not implemented at the moment." )
            return -1
        return 0

    def calc_sf_indizes_from_offsets( self, sf ):
        if self.get_offsets():
            print( "ERROR: Could not find offset files." )
            return
        
        for field in ['flty', 'slty']:
            if not field in sf.data:
                print( "ERROR: '%s' has to be from the group catalogue to use calc_sf_indizes_from_offsets()" )
                return
                
        globstart = np.zeros(len(self.loadtype), dtype='int64')
        globstart[1:] = np.cumsum(self.nparticlesall[self.loadtype[:-1]])
        
        self.data['halo']    = -np.ones( self.nparticlesall[self.loadtype].sum(), dtype='int64' )
        self.data['subhalo'] = -np.ones( self.nparticlesall[self.loadtype].sum(), dtype='int64' )
        
        part_offsets = np.zeros(len(self.loadtype), dtype='int64')
        if self.chunk is not None:
            part_offsets += self.offsets_particles[self.chunk,self.loadtype]
        
        for ptype in self.loadtype:
            pstart = part_offsets[ptype]
            pcount = self.nparticlesall[ptype]
            
            ihalos, = np.where( self.offsets_halos_particles[:,ptype] + sf.data['flty'][:,ptype] > pstart )
            if len(ihalos) > 0:
                halo    = min( ihalos )            
                while halo < sf.totngroups:
                    halo_start = self.offsets_halos_particles[halo,ptype]
                    halo_count = sf.data['flty'][halo,ptype]
                
                    imin = max( halo_start - pstart, 0 )
                    imax = min( halo_start + halo_count - pstart, pcount ) 
                
                    #print( "HALO:", halo, halo_start, halo_count, imin, imax )
                
                    if imax <= imin:
                        halo += 1
                        continue
                
                    self.data['halo'][imin:imax] = halo
                
                    if halo_start + halo_count - pstart > pcount:
                        break
                
                    halo += 1
            
            isubhalo, = np.where( self.offsets_subhalos_particles[:,ptype] + sf.data['slty'][:,ptype] > pstart )
            if len(isubhalo) > 0:
                subhalo   = min( isubhalo )
                while subhalo < sf.totnsubgroups:
                    subhalo_start = self.offsets_subhalos_particles[subhalo,ptype]
                    subhalo_count = sf.data['slty'][subhalo,ptype]
                
                    imin = max( subhalo_start - pstart, 0 )
                    imax = min( subhalo_start + subhalo_count - pstart, pcount ) 
                
                    #print( "SUBHALO:", subhalo, imin, imax )
                
                    if imax <= imin:
                        subhalo += 1
                        continue
                
                    self.data['subhalo'][imin:imax] = subhalo
                
                    if subhalo_start + subhalo_count - pstart > pcount:
                        break
                
                    subhalo += 1
        
        return

    def calc_sf_indizes( self, sf, verbose=False, halolist=[], dosubhalos=True, absolutesubnum=False, tracertype=None ):
        for field in ['flty']:
            if not field in sf.data:
                print( "ERROR: '%s' has to be from the group catalogue to use calc_sf_indizes()" % field )
                return
        
        if dosubhalos:
            for field in ['fnsh', 'ffsh', 'slty']:
                if not field in sf.data:
                    print( "ERROR: '%s' has to be from the group catalogue to use calc_sf_indizes( dosubhalos=True )" % field )
                    return
        
        if verbose:
            print("VERBOSE: Total particles per type:", self.nparticlesall)
            print("VERBOSE: Ngroups=%d, Nsubgroups=%d." % (sf.ngroups, sf.nsubgroups) )

        self.data['halo'] = -np.ones( self.nparticlesall[self.loadtype].sum(), dtype='int64' )

        globstart = np.zeros(len(self.loadtype), dtype='int64')
        globstart[1:] = np.cumsum(self.nparticlesall[self.loadtype[:-1]])
        
        globoffset = np.zeros((sf.totngroups, len(self.loadtype)), dtype='int64').astype('int64')
        if sf.totngroups > 1:
            globoffset[1:,:] = np.cumsum(sf.data['flty'][:-1,self.loadtype], axis=0)

        if dosubhalos:
            self.data['subhalo'] = -np.ones( self.nparticlesall[self.loadtype].sum(), dtype='int64' )
            
        if self.loadonlyhalo >= 0:
            halolist = [self.loadonlyhalo]
            globoffset[:,:] -= globoffset[self.loadonlyhalo,:]
            subhalocount = sf.data['fnsh'][:self.loadonlyhalo].sum()

        if not halolist and self.chunk is None:
            halolist = range( sf.totngroups )
            if verbose:
                print( "VERBOSE: halolist=",halolist )
        
        for halo in halolist:
            start = globstart[:].astype('int64') + globoffset[halo,:]
            length = sf.data['flty'][halo,self.loadtype].astype('int64')
            
            if verbose:
                print( "VERBOSE: Halo %d of %d." % (halo + 1, len(halolist)) )
                print( "VERBOSE: Start ", start, ", Length", length )
            
            for ptype in range(len(self.loadtype)):
                pstart = start[ptype]
                pend   = start[ptype] + length[ptype]
                
                if verbose:
                    print( "VERBOSE: ptype %d, start=%d, end=%d" % (self.loadtype[ptype], pstart, pend) )
                
                if pstart < globstart[ptype]:
                    pstart = globstart[ptype]
                
                if pstart >= globstart[ptype] + self.nparticlesall[self.loadtype[ptype]]:
                    continue
                
                if pend < globstart[ptype]:
                    continue
                
                if pend >= globstart[ptype] + self.nparticlesall[self.loadtype[ptype]]:
                    pend = globstart[ptype] + self.nparticlesall[self.loadtype[ptype]].astype('int64')
                
                self.data['halo'][pstart:pend] = halo

            if dosubhalos:
                startsub = start
                
                for subhalo in range( sf.data['fnsh'][halo] ):
                    lengthsub = sf.data['slty'][sf.data['ffsh'][halo] + subhalo,self.loadtype]
                
                    for ptype in range(len(self.loadtype)):
                        if tracertype is not None and self.loadtype[ptype] == tracertype:
                            continue

                        pstart = startsub[ptype]
                        pend   = startsub[ptype] + lengthsub[ptype]

                        if verbose:
                            print( "subhalo:", subhalo, "ptype:", ptype, "pstart:", pstart, "pend:", pend )
                        
                        if pstart < globstart[ptype]:
                            pstart = globstart[ptype]
                
                        if pstart >= globstart[ptype] + self.nparticlesall[self.loadtype[ptype]]:
                            continue
                
                        if pend < globstart[ptype]:
                            continue
                
                        if pend >= globstart[ptype] + self.nparticlesall[self.loadtype[ptype]]:
                            pend = globstart[ptype] + self.nparticlesall[self.loadtype[ptype]].astype('int64')
                        
                        if pend <= pstart:
                            continue
                        
                        if verbose:
                            print( "subhalo setting:", subhalo, "ptype:", ptype, "pstart:", pstart, "pend:", pend )
                        
                        if absolutesubnum:
                            self.data['subhalo'][pstart:pend] = sf.data['ffsh'][halo] + subhalo
                        else:
                            self.data['subhalo'][pstart:pend] = subhalo
                    
                    startsub += lengthsub
        return

    def set_center( self, center ):
        if type( center ) == list:
            self.center = pylab.array( center )
        elif type( center ) == np.ndarray:
            self.center = center
        else:
            raise Exception( "center has to be of type list or numpy.ndarray" )
        return
        
    def r( self, center=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        radius = pylab.sqrt( (self.data[ "pos" ][:,0]-center[0])**2+(self.data[ "pos" ][:,1]-center[1])**2+(self.data[ "pos" ][:,2]-center[2])**2 )
        return radius

    def rid( self ):
        rids = pylab.zeros( self.npart+1, dtype='int32' )
        rids[0] = -1
        rids[1:] = self.data['id'].argsort()
        return rids

    def get_principal_axis( self, idx, L=None ):
        tensor = pylab.zeros( (3,3) )

        mass = self.data['mass'][idx]
        px = self.pos[idx,0]
        py = self.pos[idx,1]
        pz = self.pos[idx,2]
        
        tensor[0,0] = (mass * (py*py + pz*pz)).sum()
        tensor[1,1] = (mass * (px*px + pz*pz)).sum()
        tensor[2,2] = (mass * (px*px + py*py)).sum()

        tensor[0,1] = - (mass * px * py).sum()
        tensor[1,0] = tensor[0,1]
        tensor[0,2] = - (mass * px * pz).sum()
        tensor[2,0] = tensor[0,2]
        tensor[1,2] = - (mass * py * pz).sum()
        tensor[2,1] = tensor[1,2]

        eigval, eigvec = scipy.linalg.eig( tensor )

        if L is None:
            maxval = eigval.argsort()[-1]
            return eigvec[:,maxval]
        else:
            A1 = (L * eigvec[:,0]).sum()
            A2 = (L * eigvec[:,1]).sum()
            A3 = (L * eigvec[:,2]).sum()

            A = np.abs( np.array( [A1, A2, A3] ) )
            i, = np.where( A == A.max() )
            xdir = eigvec[:,i[0]]

            if (xdir * L).sum() < 0:
                xdir *= -1.0

            j, = np.where( A != A.max() )
            i2 = eigval[j].argsort()
            ydir = eigvec[:,j[i2[1]]]
            
            if ydir[0] < 0:
                ydir *= -1.0

            zdir = np.cross( xdir, ydir )

            return xdir, ydir, zdir
    
    def centerat( self, center, periodic=True ):
        self.data['pos'] = self.data['pos'].astype('f8')
        self.data['pos'] -= center.astype('f8')
        
        if periodic:
            i = np.where( self.data['pos'] > +0.5 * self.boxsize )
            self.data['pos'][i] -= self.boxsize * self.time / self.hubbleparam
            i = np.where( self.data['pos'] < -0.5 * self.boxsize )
            self.data['pos'][i] += self.boxsize * self.time / self.hubbleparam
        
        #self.data['pos'] -= pylab.array( center )[None,:]
        self.center = pylab.zeros( 3 )
        return

    def centersubhaloes( self, center, sf, periodic=True):
        sf.data['spos'] = sf.data['spos'].astype('f8')
        print('sub pos before=',sf.data['spos'])
        sf.data['spos'] -= center.astype('f8')
        print('sub pos after=',sf.data['spos'])
        
        if periodic:
            i = np.where( sf.data['spos'] > +0.5 * self.boxsize )
            sf.data['spos'][i] -= self.boxsize * self.time / self.hubbleparam
            i = np.where( sf.data['spos'] < -0.5 * self.boxsize )
            sf.data['spos'][i] += self.boxsize * self.time / self.hubbleparam

        return
            
    def centerofmass( self ):
        #if 'cmce' in self.data:
            #rcm = [(self.data['cmce'][:,i]/self.data['mass'].sum()*self.data['mass']).sum() for i in [0,1,2]]
        #else:
            #rcm = [(self.pos[:,i]/self.data['mass'].sum()*self.data['mass']).sum() for i in [0,1,2]]
        rcm = [(self.pos[:,i]/self.data['mass'].sum()*self.data['mass']).sum() for i in [0,1,2]]
        return pylab.array( rcm )
    
    def rotateto( self, dir, dir2=None, dir3=None, sf=None, verbose=False ):
        if dir2 is None or dir3 is None:
            # get normals
            dir2 = pylab.zeros( 3 )
            if dir[0] != 0 and dir[1] != 0:
                dir2[0] = -dir[1]
                dir2[1] = dir[0]
            else:
                dir2[0] = 1
            dir2 /= sqrt( (dir2**2).sum() )
            dir3 = np.cross( dir, dir2 )

        matrix = np.array( [dir,dir2,dir3] )

        for value in self.data:
            if value == 'bhts' or value == 'bpos' or value == 'bvel' or value == 'bdens':   # do not rotate the BH time step field
                continue
            if 'pass' in value:
                continue
            if self.data[value] is not None and self.data[value].ndim == 2 and pylab.shape( self.data[value] )[1] == 3:
                self.rotate_value( value, matrix )
                if verbose and not self.quiet:
                    print("Rotated %s." % value)

        if sf:
            for value in sf.data:
                if sf.data[value] is not None and sf.data[value].ndim == 2 and pylab.shape( sf.data[value] )[1] == 3:
                    self.rotate_value_sf( sf, value, matrix )
        self.convenience()
        return
    
    def rotate_value( self, value, matrix ):
        rotmat = np.array( matrix.transpose(), dtype=self.data[value].dtype )
        self.data[value] = np.dot(self.data[value], rotmat)
        return

    def rotate_value_sf( self, sf, value, matrix ):
        print('Rotating subhaloes...')
        rotmat = np.array( matrix.transpose(), dtype=sf.data[value].dtype )
        print('sub values=',sf.data[value])
        sf.data[value] = np.dot(sf.data[value], rotmat)
        print('sub values after=',sf.data[value])
        return

    def select_halo( self, sf, age_select=None, haloid=0, centre=[], remove_bulk_vel=True, galradfac=0.1, rotate_disk=True, do_rotation=False, use_principal_axis=True, euler_rotation=False, align_spin_axis=False, use_cold_gas_spin=False, verbose=True, subhalo=None, centersubhaloes=False ):

        if subhalo: # warning! when subhalo is found from merger tree, calc_sf_indizes() (called before this method) must have absolutesubnum keyword enabled
            center = sf.data['spos'][subhalo,:].astype('f8')
        elif centre:
            center = np.array(centre)
        else:
            center = sf.data['fpos'][haloid,:].astype('f8')

        galrad = np.maximum(galradfac * sf.data['frc2'][haloid], 0.01)
        print('galrad=',galrad)
        self.centerat( center )
        if centersubhaloes:
            print('Centering Subhaloes...')
            self.centersubhaloes( center, sf )

            
        if remove_bulk_vel:
            iall, = np.where( self.r() < galrad )
            mass = self.data['mass']
            vel = np.sum( self.data['vel'][iall,:]*mass[iall][:,None], axis=0, dtype=np.float64 ) / np.sum( mass[iall].astype('float64'), dtype=np.float64 )
            self.data['vel'] -= vel[None,:]
            #CAUTION this is only to plot arrrows in the satellite plot!!!
            sf.data['svel'] -= vel[None,:]
            #print('sub pos=',sf.data['spos'])
            
        if rotate_disk:
            if haloid < sf.data['fnsh'][0]:
                if age_select is not None:
                    if not self.quiet:
                        print( "!!! agesel" )
                    st = self.nparticlesall[:4].sum()
                    en = st + self.nparticlesall[4]
                    age = np.zeros(self.npartall)
                    age[st:en] = self.data['age']
                    if subhalo is not None:
                        istars, = np.where( (self.r() < galrad) & (self.type == 4) & (age > 0.) & (self.data['subhalo'] == subhalo) )
                    elif centre: # do not want haloid constraint if setting centre by hand
                        istars, = np.where( (self.r() < 0.01) & (self.type == 4) & (age > 0.) )
                    else:
                        istars, = np.where( (self.r() < 0.01) & (self.type == 4) & (age > 0.) & (self.data['halo'] == haloid) )
                    ages = self.cosmology_get_lookback_time_from_a( age[istars], is_flat=True )
                    ts = self.cosmology_get_lookback_time_from_a( self.time, is_flat=True )
                    jj, = np.where( ( (ages - ts) < age_select ) )
                    istars = istars[jj]
                else:
                    if subhalo is not None:
                        istars, = np.where( (self.r() < galrad) & (self.type == 4) & (self.data['halo'] == haloid) & (self.data['subhalo'] == subhalo) )
                    elif centre:
                        istars, = np.where( (self.r() < galrad) & (self.r() > 0.) & (self.type == 4) )
                    else:
                        istars, = np.where( (self.r() < galrad) & (self.r() > 0.) & (self.type == 4) & (self.data['halo'] == haloid) )
            else: # if desired halo particles are in the fuzz, do away with haloid criter.
                istars, = np.where( (self.r() < galrad) & (self.type == 4))

            if not self.quiet:
                print("Found %d stars." % np.size(istars))
            mass = self.data['mass']
            L = np.cross( self.pos[istars,:].astype('float64'), (self.vel[istars,:].astype('float64') * mass[istars][:,None].astype('float64')) )
            Ltot = L.sum( axis=0 )
            Ldir = Ltot / sqrt( (Ltot**2).sum() )

            if use_principal_axis:
                xdir, ydir, zdir = self.get_principal_axis( istars, L=Ldir )
                if do_rotation:
                    if centersubhaloes:
                        self.rotateto( xdir, dir2=ydir, dir3=zdir, sf=sf, verbose=verbose )
                    else:
                        self.rotateto( xdir, dir2=ydir, dir3=zdir, verbose=verbose )
                return np.array( [xdir, ydir, zdir] )
            elif align_spin_axis: #untested!
                zvec = np.array([1., 0., 0.]).astype('float64')
                v = np.cross(Ldir, zvec)
                c = np.dot(Ldir, zvec)
                s = np.linalg.norm(v)
                I = np.identity(3)
                k = np.array( [[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]] ).T
                r = I + k + np.matmul(k,k) * ((1 -c)/(s**2))
                
                matrix = np.array( r, dtype=self.data['pos'].dtype )

                self.data['pos'] = np.dot( self.data['pos'], matrix )
                self.data['vel'] = np.dot( self.data['vel'], matrix )

            elif euler_rotation: 
                #jj, =  np.where(age > 0.)
                pos = np.fliplr( self.data['pos'] ) # we will work in (x,y,z) instead of the usual (z,y,x)
                vel = np.fliplr( self.data['vel'] )
                spos = np.fliplr( self.data['pos'][self.data['type']==4] )
                svel = np.fliplr( self.data['vel'][self.data['type']==4] )

                sr = np.linalg.norm(spos, axis=1)
                ind = (self.data['age'].astype('float64')>0.) & (sr < 0.01)
                l = np.cross(spos[ind,:].astype('float64'), svel[ind,:].astype('float64'))

                ltot = l.sum( axis=0 )
                Ldir = ltot / sqrt( (ltot**2).sum() )
                zvec = np.array([0., 0., np.sign(Ldir[1])]).astype('float64') # set current coord vecs
                yvec = np.array([0., 1., 0.]).astype('float64')
                lon = np.cross( zvec, Ldir ) 
                lon /= sqrt(((lon**2).sum()))

                beta = np.arccos( np.dot(zvec, Ldir) ) # angle between z coord and spin axis
                alpha = np.arccos( np.dot(yvec, lon) ) # angle between y coord and plane perp to spin and z coord

                # write the rotation matrices
                rotz = np.array( [ [np.cos(alpha), -np.sin(alpha), 0.], [np.sin(alpha), np.cos(alpha), 0.], [0., 0., 1.] ] ).astype('float64')
                roty = np.array( [ [np.cos(beta), 0., np.sin(beta)], [0., 1., 0.], [-np.sin(beta), 0., np.cos(beta)] ] ).astype('float64')
                rotx = np.cross( roty, rotz ).astype('float64')

                am_rotation = np.matmul(rotz, roty)
                
                # rotate position and velocity and convert back to (z,y,x)
                tpos = np.dot( pos, am_rotation )
                tvel = np.dot( vel, am_rotation )      
                self.data['pos'] = np.fliplr(tpos)
                self.data['vel'] = np.fliplr(tvel)
                if centersubhaloes:
                    tpos = np.dot( np.fliplr(sf.data['spos']), am_rotation )
                    tvel = np.dot( np.fliplr(sf.data['svel']), am_rotation )
                    sf.data['spos'] = np.fliplr(tpos)
                    sf.data['svel'] = np.fliplr(tvel)

                return ltot, np.array( [rotx, roty, rotz] )

            elif use_cold_gas_spin:
                igas, = np.where( (self.r() < galrad) & (self.type == 0) & (self.data['subhalo'] == haloid) )
                index = self.data['sfr'][igas] > 0.
                if not self.quiet:
                    print("Found %d star forming gas particles." % np.size(igas[index]))
                mass = self.data['mass'].astype('float64')
                L = np.cross( self.pos[igas[index],:].astype('float64'), (self.vel[igas[index],:].astype('float64') * mass[igas[index]][:,None]) )
                Ltot = L.sum( axis=0 )
                Ldir = Ltot / sqrt( (Ltot**2).sum() )

                xdir, ydir, zdir = self.get_principal_axis( igas[index], L=Ldir )

                if do_rotation:
                    if centersubhaloes:
                        self.rotateto( xdir, dir2=ydir, dir3=zdir, sf=sf, verbose=verbose )
                    else:
                        self.rotateto( xdir, dir2=ydir, dir3=zdir, verbose=verbose )
                return np.array( [xdir, ydir, zdir] )
            else:
                dir = Ldir
                if do_rotation:
                    if centersubhaloes:
                        self.rotateto( dir, sf=sf, verbose=verbose )
                    else:
                        self.rotateto( dir, verbose=verbose )
                return dir
            return
        return False

    def get_slice( self, value, box=[0,0], nx=200, ny=200, center=False, axes=[0,1] ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        dim0 = axes[0]
        dim1 = axes[1]
        
        if (box[0] == 0 and box[1] == 0):
            box[0] = max( abs( self.data[ "pos" ][:,dim0] ) ) * 2
            box[1] = max( abs( self.data[ "pos" ][:,dim1] ) ) * 2

        if (value == "mass"):
            return calcGrid.calcDensSlice( self.data["pos"].astype('float64'), self.data["hsml"].astype('float64'), self.data[value].astype('float64'), nx, ny, box[0], box[1], center[0], center[1], center[2], dim0, dim1 )
        else:
            return calcGrid.calcSlice( self.data["pos"].astype('float64'), self.data["hsml"].astype('float64'), self.data["mass"].astype('float64'), self.data["rho"].astype('float64'), self.data[value].astype('float64'), nx, ny, box[0], box[1], center[0], center[1], center[2], dim0, dim1 )

    def get_raddens( self, nshells=200, dr=0, center=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        return calcGrid.calcRadialProfile( self.pos.astype('float64'), self.data["mass"].astype('float64'), 1, nshells, dr, center[0], center[1], center[2] )

    def get_radprof( self, value, nshells=200, dr=0, center=False, mode=2 ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        if 'type' in self.data and value != 'angmom':
            indgas = self.data['type'] == 0
            return calcGrid.calcRadialProfile( self.pos.astype('float64')[indgas], self.data[value].astype('float64'), mode, nshells, dr, center[0], center[1], center[2] )
        return calcGrid.calcRadialProfile( self.pos.astype('float64'), self.data[value].astype('float64'), mode, nshells, dr, center[0], center[1], center[2] )
    
    def get_radmassprof( self, nshells, dr=0, center=False, solarmass=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center        

        p = calcGrid.calcRadialProfile( self.pos.astype('float64'), self.data["mass"].astype('float64'), 0, nshells, dr, center[0], center[1], center[2] )
        for i in range( 1, nshells ):
            p[0,i] += p[0,i-1]
        if solarmass:
            p[0,:] /= 1.989e33
        return p

    def plot_raddens( self, log=False, nshells=200, dr=0, center=False, color='k' ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center    
        
        p = self.get_raddens( nshells, dr, center )
        if log:
            pylab.semilogy( p[1,:], p[0,:], color )
        else:
            pylab.plot( p[1,:], p[0,:], color )

    def plot_radprof( self, value, log=False, nshells=200, dr=0, center=False, color='k', mode=2 ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center    
        
        p = self.get_radprof( value, nshells, dr, center, mode=mode )
        if log:
            pylab.semilogy( p[1,:], p[0,:], color )
        else:
            pylab.plot( p[1,:], p[0,:], color )

    def plot_radmassprof( self, log=False, nshells=200, dr=0, center=False, color='k', solarmass=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        p = self.get_radmassprof( nshells, dr, center, solarmass )
        if log:
            pylab.semilogy( p[1,:], p[0,:], color )
        else:
            pylab.plot( p[1,:], p[0,:], color )
    
    def plot_radvecprof( self, value, log=False, nshells=200, dr=0, center=False, color='k' ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        vec = (self.data[value]*self.data['pos']).sum(axis=1)/self.r()
        p = calcGrid.calcRadialProfile( self.pos.astype('float64'), vec.astype('float64'), 2, nshells, dr, center[0], center[1], center[2] )
        if log:
            pylab.semilogy( p[1,:], p[0,:], color )
        else:
            pylab.plot( p[1,:], p[0,:], color )

    def plot_pos( self, axes=[0,1] ):
        pylab.plot( self.pos[:,axes[0]], self.pos[:,axes[1]], ',' )
        pylab.axis( "scaled" )
        
    def print_abundances( self, species = None ):
        sp = None
        if species is not None:
          sp = loaders.load_species( species )

          if sp['count'] != self.nspecies:
            print("Number of species in speciesfile (%d) and snapshot (%d) don't match." % (sp['count'],self.nspecies))
            sp = None
        
        print("Total abundances in solar masses:")
        for i in range( self.nspecies ):
            if sp is not None:
              print("Species %d (%s): %g" % (i,sp['names'][i],(self.data['mass'][:self.nparticlesall[0]].astype('float64') * self.data['xnuc'][:,i]).sum()/msol))
            else:
              print("Species %d: %g" % (i,(self.data['mass'][:self.nparticlesall[0]].astype('float64') * self.data['xnuc'][:,i]).sum()/msol))
        return

    def plot_slice( self, value, logplot=True, colorbar=False, box=[0,0], nx=200, ny=200, center=False, axes=[0,1], minimum=1e-8, newfig=True, nolabels=False, cmap=False, vrange=False, rasterized=True, cblabel=False, logfm=True ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        dim0 = axes[0]
        dim1 = axes[1]
        
        if (box[0] == 0 and box[1] == 0):
            box[0] = max( abs( self.data[ "pos" ][:,dim0] ) ) * 2
            box[1] = max( abs( self.data[ "pos" ][:,dim1] ) ) * 2

        slice = self.get_slice( value, box, nx, ny, center, axes )
        x = (pylab.array( range( nx+1 ) ) - nx/2.) / nx * box[0]
        y = (pylab.array( range( ny+1 ) ) - ny/2.) / ny * box[1]

        if newfig:
            fig = pylab.figure( figsize = ( 13, int(12*box[1]/box[0] + 0.5) ) )
            
        if logplot:
            slice = pylab.maximum( slice, minimum )

        if not vrange:
            vrange = [ slice.min(), slice.max() ]

        if logplot:
            if logfm:
                pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized )
            else:
                pc = pylab.pcolormesh( x, y, pylab.transpose( pylab.log10(slice) ), shading='flat', rasterized=rasterized )
        else:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized )
        
        if cmap:
            pylab.set_cmap( cmap )

        if colorbar:
            if logplot:
                cb = pylab.colorbar( format=matplotlib.ticker.LogFormatterMathtext() )
            else:
                fmt = matplotlib.ticker.ScalarFormatter( useMathText=True )
                fmt.set_powerlimits( (-2, 2) )
                fmt.set_useOffset( False )
                cb = pylab.colorbar( format=fmt )
            if cblabel:
                    cb.set_label( cblabel )
        
        pylab.axis( "image" )

        if not nolabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=16, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=16, ha='right' )
        return pc

    def plot_cylav( self, value, logplot=True, box=[0,0], nx=512, ny=512, center=False, minimum=1e-8 ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        if (box[0] == 0 and box[1] == 0):
            box[0] = max( abs( self.data[ "pos" ][:,0] ) ) * 2
            box[1] = max( abs( self.data[ "pos" ][:,1:] ) ) * 2

        grid = calcGrid.calcGrid( self.pos.astype('float64'), self.data["hsml"].astype('float64'), self.data["mass"].astype('float64'), self.data["rho"].astype('float64'), self.data[value].astype('float64').astype('float64'), nx, ny, ny, box[0], box[1], box[1], 0, 0, 0 )
        cylav = calcGrid.calcCylinderAverage( grid )
        x = (pylab.array( range( nx+1 ) ) - nx/2.) / nx * box[0]
        y = (pylab.array( range( ny+1 ) ) - ny/2.) / ny * box[1]

        fig = pylab.figure( figsize = ( 13, int(12*box[1]/box[0] + 0.5) ) )
        pylab.spectral()
        
        if logplot:
            pc = pylab.pcolor( x, y, pylab.transpose( pylab.log10( pylab.maximum( cylav, minimum ) ) ), shading='flat' )
        else:
            pc = pylab.pcolor( x, y, pylab.transpose( slice ), shading='flat' )

        pylab.axis( "image" )
        xticklabels = []
        for tick in pc.axes.get_xticks():
            if (tick == 0):
                xticklabels += [ r'$0.0$' ]
            else:
                xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
        pc.axes.set_xticklabels( xticklabels, size=16, y=-0.1, va='baseline' )

        yticklabels = []
        for tick in pc.axes.get_yticks():
            if (tick == 0):
                yticklabels += [ r'$0.0$' ]
            else:
                yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
        pc.axes.set_yticklabels( yticklabels, size=16, ha='right' )
        return pc

    def getbound( self, center=False, vel=[0,0,0] ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        start = time.time()
        radius = pylab.zeros( self.npartall )
        for i in range( 3 ):
            radius += (self.data[ "pos" ][:,i] - center[i])**2
        radius = pylab.sqrt( radius )
        rs = radius.argsort()
        
        mass = 0.
        bcount = 0.
        bmass = 0.
        bcenter = [0., 0., 0.]
        bparticles = []
        for part in range( self.npart ):
            if (part == 0) or (( ( self.vel[rs[part],:] - vel )**2. ).sum() < 2.*G*mass/radius[rs[part]]):
                bcount += 1.
                bmass += self.data['mass'][rs[part]]
                bcenter += self.pos[rs[part],:]
                bparticles += [self.id[rs[part]]]
            mass += self.data['mass'][rs[part]]
        
        bobject = {}
        bobject['mass'] = bmass
        bobject['center'] = bcenter / bcount
        bobject['count'] = bcount
        bobject['particles'] = bparticles
        
        print("Calculation took %gs." % (time.time()-start))
        return bobject

    def mapOnCartGrid( self, value, center=False, box=False, res=512, saveas=False, use_only_cells=None, numthreads=1):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize,self.boxsize] )

        if type( res ) == list:
            res = pylab.array( res )
        elif type( res ) != np.ndarray:
            res = np.array( [res]*3 )
            
        if use_only_cells is None:
            use_only_cells = np.arange( self.nparticlesall[0], dtype='int32' )

        pos = self.pos[use_only_cells,:].astype( 'float64' )
        px = np.abs( pos[:,0] - center[0] )
        py = np.abs( pos[:,1] - center[1] )
        pz = np.abs( pos[:,2] - center[2] )

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < 0.5*box[2]) )
        print("Selected %d of %d particles." % (pp.size,self.npart))

        posdata = pos[pp]
        valdata = self.data[value][use_only_cells][pp].astype('float64')
        
        if valdata.ndim == 1:
            data = calcGrid.calcASlice(posdata, valdata, nx=res[0], ny=res[1], nz=res[2], boxx=box[0], boxy=box[1], boxz=box[2],
                                       centerx=center[0], centery=center[1], centerz=center[2], grid3D=True, numthreads=numthreads)
            grid = data["grid"]
        else:
            # We are going to generate ndim 3D grids and stack them together
            # in a grid of shape (valdata.shape[1],res,res,res)
            grid = []
            for dim in range(valdata.shape[1]):
                data = calcGrid.calcASlice(posdata, valdata[:,dim], nx=res[0], ny=res[1], nz=res[2], boxx=box[0], boxy=box[1], boxz=box[2],
                                           centerx=center[0], centery=center[1], centerz=center[2], grid3D=True, numthreads=numthreads)
                grid.append(data["grid"])
            grid = np.stack([subgrid for subgrid in grid])
        if saveas:
            grid.tofile( saveas )

        return grid

    def get_Aslice( self, value, grad=False, res=1024, center=False, axes=[0,1], proj_fact=0.5, box=False, proj=False, polar=False, nx=None, ny=None, nz=None, boxz=None, allsky=False, allsky_phi_offset=0, numthreads=1, tree=None, keepTree=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        if 'type' in self.data:
            pos = self.pos.astype( 'float64' )[self.data['type'] == 0]
        else:
            pos = self.pos.astype( 'float64' )

        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        if nz is None:
          nz = int(2*proj_fact*res)
        
        if proj:
            if boxz is not None:
              zdist = 0.5 * boxz
            else:
              if box.shape[0] == 3:
                zdist = proj_fact * box[2]
              else:
                zdist = proj_fact * box.max()
              boxz = 2.*zdist
        else:
          boxz = 0
          zdist = 2. * self.data['vol'].astype('float64')**(1./3.)

        if polar:
          rad = np.sqrt( px*px + py*py )
          pp, = np.where( (rad < box[0]) & (pz < zdist) )
        elif allsky:
          rad = np.sqrt( px*px + py*py + pz*pz )
          pp, = np.where( rad < box[0] )
        else:
          pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )
        print("Selected %d of %d particles." % (pp.size,self.npart))

        if proj:
                print( "nz=", nz, "zdist=%g, boxx=%f, boxy=%g" % (zdist, box[0], box[1]) )
        else:
                print( "nz=", nz, "zdist=%g, boxx=%f, boxy=%g" % (zdist.min(), box[0], box[1]) )


        if nx is None:
          nx = res
        if ny is None:
          ny = res

        posdata = pos[pp,:]
        valdata = self.data[value][pp].astype('float64')
        
        kwargs = {}
        if grad:
          kwargs['grad'] = graddata = self.data[grad][pp,:].astype('float64')
        if tree:
          kwargs['tree'] = tree

        data = calcGrid.calcASlice( posdata, valdata, nx, ny, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads, polar=polar, allsky=allsky, allsky_phi_offset=allsky_phi_offset, keepTree=keepTree, **kwargs )
        slice = data[ "grid"]

        print("Total is ", slice.sum() * box.max() / res)

        if not polar:
          data['x'] = pylab.arange( nx+1, dtype="float64" ) / nx * box[0] - .5 * box[0] + c[0]
          data['y'] = pylab.arange( ny+1, dtype="float64" ) / ny * box[1] - .5 * box[1] + c[1]
        else:
          data['r'] = pylab.arange( nx+1, dtype="float64" ) / nx * box[0]
          data['phi'] = pylab.arange( ny+1, dtype="float64" ) / ny * 2.*pi

        if not proj and not allsky:
                data[ "neighbours" ] = pp[ data["neighbours"] ]

        return data

    def save_Aslice( self, value, fname, path='', flat=False, weight=False, grad=False, res=1024, center=False, axes=[0,1], proj_fact=0.5, box=False, proj=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        if 'type' in self.data:
            pos = self.pos.astype( 'float64' )[self.data['type'] == 0]
        else:
            pos = self.pos.astype( 'float64' )

        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        nz = int(2*proj_fact*res)
        boxz = 0

        zdist = 2. * self.data['vol'].astype('float64')**(1./3.)
        if proj:
            if box.shape[0] == 3:
              zdist = proj_fact * box[2]
            else:
              zdist = proj_fact * box.max()

            boxz = 2. * zdist

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )
        print( "Selected %d of %d particles." % (pp.size,self.npart) )
        print( "nz=", nz, "zdist=%g, boxx=%f, boxy=%g" % (zdist, box[0], box[1]) )

        posdata = pos[pp,:]
        valdata = self.data[value][pp].astype('float64')

        if weight:
                weightdata = self.data[weight][pp].astype('float64')
                

        print( "Total should be ", valdata.sum() )

        if not grad:
                data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz )
                if weight:
                        wdata = calcGrid.calcASlice( posdata, weightdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz )
        else:
                graddata = self.data[grad][pp,:].astype('float64')
                data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, graddata, proj=proj, boxz=boxz, nz=nz )
                if weight:
                        wdata = calcGrid.calcASlice( posdata, weightdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, graddata, proj=proj, boxz=boxz, nz=nz )

        if weight:
                slice = data["grid"] / wdata["grid"]
        else:
                slice = data["grid"]

        if flat:
                filename = path + fname + '-flat'
                np.save( filename, slice.flatten() )
        else:
                filename = path + fname
                np.save( filename, slice )

    def load_Aslice( self, fname, path ):
        filename = path + fname
        return np.load( filename )

    def plot_Aslice( self, value, grad=False, logplot=False, divzero=False, divzero_centre=0, colorbar=False, contour=False, res=1024, center=False, axes=[0,1], proj_fact=0.5, minimum=1e-8, newfig=True, newlabels=False, cmap=False, vrange=False, cblabel=False, rasterized=False, box=False, proj=False, dextoshow=False, neglog=False, cmap2=False, levels=False, numthreads=1, pixel_scale =1 ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
                box = pylab.array( box )
        elif type( box ) != np.ndarray:
                box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = self.data['pos'][:self.nparticlesall[0]].astype( 'float64' )

        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        nz = int(2*proj_fact*res)
        boxz = 0

        if proj:
            if box.shape[0] == 3:
              zdist = proj_fact * box[2]
            else:
              zdist = proj_fact * box.max()

            boxz = 2. * zdist
        else:
            zdist = 2. * self.data['vol'].astype('float64')**(1./3.)

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )
        print( "Selected %d of %d particles." % (pp.size,self.npart) )
        if pp.size == 0:
            print("No particles found, exiting.")
            return

        posdata = pos[pp,:]
        valdata = self.data[value][pp].astype('float64')
        if not grad:
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
        else:
            graddata = self.data[grad][pp,:].astype('float64')
            if 'cmce' in self.data:
                pcenter = self.data['cmce'][pp,:].astype('float64')
                data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, grad=graddata, pcenter=pcenter, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
            else:
                data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, grad=graddata, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
        slice = data[ "grid"]

        print('\t>>Adjusting for pixel scale...')
        slice /= pixel_scale

        if (not proj):
            neighbours = data[ "neighbours" ]
            contours = data[ "contours" ]
        x = pylab.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        y = pylab.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]

        if newfig:
            fig = pylab.figure()

        if logplot:
            if neglog:
                sliceneg = -slice.copy()

                index = (slice <= 0.0)
                slice = pylab.maximum( slice, minimum )
                slice[index] = 0.1 * minimum

                index = (sliceneg < 0.0)
                sliceneg = pylab.maximum( sliceneg, minimum )
                sliceneg[index] = 0.1 * minimum
            else:
                slice = pylab.maximum( slice, minimum )

            if dextoshow:
                if not vrange:
                    if np.log10(slice.max() / slice.min()) > dextoshow:
                        vrange = [ slice.max() / 10.0**dextoshow, slice.max() ]
                else:
                    print( "vrange already chosen so dextoshow is not applied" )

        if not logplot:
            print( "dextoshow is not supported for non logarithmic plots" )

        if not vrange:
            vrange = [ slice.min(), slice.max() ]
            vrange[0] = pylab.maximum(vrange[0], minimum)
            print( "Plot range ", vrange )

        if cmap:
            #pylab.set_cmap( cmap )
            #cmapref = pylab.get_cmap()
            cmapref = cmap
            if neglog:
                cmapref.set_under('white', alpha=0)
                if cmap2:
                   pylab.set_cmap( cmap2 )
                   cmapref2 = pylab.get_cmap()
                   cmapref2.set_under('white', alpha=0)
                   #pylab.set_cmap( cmapref )
        else:
                cmapref = pylab.get_cmap()
                        
        
        if logplot:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized, cmap=cmapref)
            if neglog:
                pc = pylab.pcolormesh( x, y, pylab.transpose( sliceneg ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized, cmap=cmapref2 )
        else:
            if divzero: #added 13/01/23
                pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.TwoSlopeNorm(vmin=vrange[0], vmax=vrange[1], vcenter=divzero_centre), rasterized=rasterized, cmap=cmapref)
            else:
                pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized, vmin=vrange[0], vmax=vrange[1], cmap=cmapref )
        
        if colorbar:
            if logplot:
                cb = pylab.colorbar( format=matplotlib.ticker.LogFormatterMathtext() )
            else:
                fmt = matplotlib.ticker.ScalarFormatter( useMathText=True )
                fmt.set_powerlimits( (-3, 3) )
                fmt.set_useOffset( False )
                cb = pylab.colorbar( format=fmt )
            if cblabel:
                cb.set_label( cblabel )
        
        if contour:
            x = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[0] - .5 * box[0] + center[0]   
            y = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[1] - .5 * box[1] + center[1]
            if not proj:
                    pylab.contour( x, y, pylab.transpose( contours ), levels=[0.99], colors="w" )
            else:
                    pylab.contour( x, y, pylab.transpose( slice ), levels=levels, colors="k", linewidths=0.4 )

        pylab.axis( "image" )

        if newlabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=24, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=24, ha='right' )
        return pc


    def axplot_Aslice( self, ax, value, grad=False, logplot=False, divzero=False, divzero_centre=0, colorbar=False, contour=False, res=1024, center=False, axes=[0,1], proj_fact=0.5, minimum=1e-8, newfig=True, newlabels=False, cmap=False, vrange=False, cblabel=False, rasterized=False, box=False, proj=False, dextoshow=False, neglog=False, cmap2=False, levels=False, numthreads=1, pixel_scale =1 ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
                box = pylab.array( box )
        elif type( box ) != np.ndarray:
                box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = self.data['pos'][:self.nparticlesall[0]].astype( 'float64' )

        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        nz = int(2*proj_fact*res)
        boxz = 0

        if proj:
            if box.shape[0] == 3:
              zdist = proj_fact * box[2]
            else:
              zdist = proj_fact * box.max()

            boxz = 2. * zdist
        else:
            zdist = 2. * self.data['vol'].astype('float64')**(1./3.)

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )
        print( "Selected %d of %d particles." % (pp.size,self.npart) )
        if pp.size == 0:
            print("No particles found, exiting.")
            return

        posdata = pos[pp,:]
        valdata = self.data[value][pp].astype('float64')
        if not grad:
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
        else:
            graddata = self.data[grad][pp,:].astype('float64')
            if 'cmce' in self.data:
                pcenter = self.data['cmce'][pp,:].astype('float64')
                data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, grad=graddata, pcenter=pcenter, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
            else:
                data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, grad=graddata, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
        slice = data[ "grid"]

        print('Adjusting for pixel scale...')
        print(slice.max())
        slice /= pixel_scale
        print(slice.max())

        if (not proj):
            neighbours = data[ "neighbours" ]
            contours = data[ "contours" ]
        x = pylab.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        y = pylab.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]

        if newfig:
            fig = pylab.figure()

        if logplot:
            if neglog:
                sliceneg = -slice.copy()

                index = (slice <= 0.0)
                slice = pylab.maximum( slice, minimum )
                slice[index] = 0.1 * minimum

                index = (sliceneg < 0.0)
                sliceneg = pylab.maximum( sliceneg, minimum )
                sliceneg[index] = 0.1 * minimum
            else:
                slice = pylab.maximum( slice, minimum )

            if dextoshow:
                if not vrange:
                    if np.log10(slice.max() / slice.min()) > dextoshow:
                        vrange = [ slice.max() / 10.0**dextoshow, slice.max() ]
                else:
                    print( "vrange already chosen so dextoshow is not applied" )

        if not logplot:
            print( "dextoshow is not supported for non logarithmic plots" )

        if not vrange:
            vrange = [ slice.min(), slice.max() ]
            vrange[0] = pylab.maximum(vrange[0], minimum)
            print( "Plot range ", vrange )

        if cmap:
            #pylab.set_cmap( cmap )
            #cmapref = pylab.get_cmap()
            cmapref = cmap
            if neglog:
                cmapref.set_under('white', alpha=0)
                if cmap2:
                   pylab.set_cmap( cmap2 )
                   cmapref2 = pylab.get_cmap()
                   cmapref2.set_under('white', alpha=0)
                   #pylab.set_cmap( cmapref )
        else:
                cmapref = pylab.get_cmap()
                        
        
        if logplot:
            pc = ax.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized, cmap=cmapref)
            if neglog:
                pc = ax.pcolormesh( x, y, pylab.transpose( sliceneg ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized, cmap=cmapref2 )
        else:
            if divzero: #added 13/01/23
                pc = ax.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.TwoSlopeNorm(vmin=vrange[0], vmax=vrange[1], vcenter=divzero_centre), rasterized=rasterized, cmap=cmapref)
            else:
                pc = ax.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized, vmin=vrange[0], vmax=vrange[1], cmap=cmapref )
        
        if colorbar:
            if logplot:
                cb = ax.colorbar( format=matplotlib.ticker.LogFormatterMathtext() )
            else:
                fmt = matplotlib.ticker.ScalarFormatter( useMathText=True )
                fmt.set_powerlimits( (-3, 3) )
                fmt.set_useOffset( False )
                cb = ax.colorbar( format=fmt )
            if cblabel:
                cb.set_label( cblabel )
        
        if contour:
            x = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[0] - .5 * box[0] + center[0]   
            y = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[1] - .5 * box[1] + center[1]
            if not proj:
                    ax.contour( x, y, pylab.transpose( contours ), levels=[0.99], colors="w" )
            else:
                    ax.contour( x, y, pylab.transpose( slice ), levels=levels, colors="k", linewidths=0.4 )

        pylab.axis( "image" )

        if newlabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=24, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=24, ha='right' )
        return pc



    def plot_Aweightedslice( self, value, weights, grad=False, logplot=False, divzero=False, divzero_centre=0, colorbar=False, contour=False, res=1024, center=False, axes=[0,1], proj_fact=0.5, minimum=1e-8, newfig=True, newlabels=False, cmap=False, vrange=False, cblabel=False, rasterized=False, box=False, proj=False, dextoshow=False, noweight=False, absolute=False, levels=False, numthreads=1 , maskplot=False, bound_norm=None ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = self.data['pos'].astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        nz = int(2*res*proj_fact)
        boxz = 0

        if proj:
            if box.shape[0] == 3:
              zdist = proj_fact * box[2]
            else:
              zdist = proj_fact * box.max()

            boxz = 2.* zdist
        else:
            zdist = 2. * self.data['vol'].astype('float64')**(1./3.)

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )
        print( "Selected %d of %d particles." % (pp.size,self.npartall) )
        print( "nz=", nz, "zdist=%g, boxx=%f, boxy=%g" % (zdist.min(), box[0], box[1]) )

        posdata = pos[pp,:]
        weightdata = self.data[weights][pp].astype('float64')

        if not noweight:
                valdata = self.data[value][pp].astype('float64') * weightdata
        else:
                valdata = self.data[value][pp].astype('float64')

        if not grad:
            weightslice = calcGrid.calcASlice( posdata, weightdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
        else:
            graddata = self.data[grad][pp,:].astype('float64')
            weightslice = calcGrid.calcASlice( posdata, weightdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, graddata, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )

        if not grad:
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
        else:
            graddata = self.data[grad][pp,:].astype('float64')
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, graddata, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )

        slice = data["grid"] / weightslice["grid"]

        if absolute:
            slice = np.abs( slice )

        if (not proj):
            neighbours = data[ "neighbours" ]
            contours = data[ "contours" ]
        x = pylab.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        y = pylab.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]

        if newfig:
            fig = pylab.figure(figsize=(12,12))

        if logplot:
            slice = pylab.maximum( slice, minimum )
            if dextoshow:
                 if not vrange:
                      if np.log10(slice.max() / slice.min()) > dextoshow:
                            vrange = [ slice.max() / 10.0**dextoshow, slice.max() ]
                 else:
                      print( "vrange already chosen so dextoshow is not applied" )

        if not logplot:
            print("dextoshow is not supported for non logarithmic plots")

        if not vrange:
            vrange = [ slice.min(), slice.max() ]

        print("Plot range", vrange)
        

        if maskplot:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized, cmap=cmap )
        
        elif logplot:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized )
        
        else:
            if divzero: #added 30/05/23
                print('Divzero: ',vrange[0],vrange[1])
                pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.TwoSlopeNorm(vmin=vrange[0], vmax=vrange[1], vcenter=divzero_centre), rasterized=rasterized)#, cmap=cmapref)
            else:
                pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized, vmin=vrange[0], vmax=vrange[1])#, cmap=cmapref )
        
        #else:
        #    pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized, vmin=vrange[0], vmax=vrange[1] )
        
        if cmap and not maskplot:
            pylab.set_cmap( cmap )

        if colorbar:
            if logplot:
                cb = pylab.colorbar( format=matplotlib.ticker.LogFormatterMathtext() )
            else:
                fmt = matplotlib.ticker.ScalarFormatter( useMathText=True )
                fmt.set_powerlimits( (-2, 2) )
                fmt.set_useOffset( False )
                cb = pylab.colorbar( format=fmt )
            if cblabel:
                cb.set_label( cblabel )
        
        if contour:
            x = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[0] - .5 * box[0] + center[0]
            y = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[1] - .5 * box[1] + center[1]
            if not proj:
                    pylab.contour( x, y, pylab.transpose( contours ), levels=[0.99], colors="w" )
            else:
                    pylab.contour( x, y, pylab.transpose( slice ), levels=levels, colors="k" )

        pylab.axis( "image" )

        if newlabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=24, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=24, ha='right' )
        return pc

    def axplot_Aweightedslice( self, ax, value, weights, grad=False, logplot=False, divzero=False, divzero_centre=0, colorbar=False, contour=False, res=1024, center=False, axes=[0,1], proj_fact=0.5, minimum=1e-8, newlabels=False, cmap=False, vrange=False, cblabel=False, rasterized=False, box=False, proj=False, dextoshow=False, noweight=False, absolute=False, levels=False, numthreads=1,maskplot=False):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = self.data['pos'].astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        nz = int(2*res*proj_fact)
        boxz = 0

        if proj:
            if box.shape[0] == 3:
              zdist = proj_fact * box[2]
            else:
              zdist = proj_fact * box.max()

            boxz = 2.* zdist
        else:
            zdist = 2. * self.data['vol'].astype('float64')**(1./3.)

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist))
        print( "Selected %d of %d particles." % (pp.size,self.npartall) )
        print( "nz=", nz, "zdist=%g, boxx=%f, boxy=%g" % (zdist.min(), box[0], box[1]) )

        posdata = pos[pp,:]
        weightdata = self.data[weights][pp].astype('float64')

        if not noweight:
                valdata = self.data[value][pp].astype('float64') * weightdata
        else:
                valdata = self.data[value][pp].astype('float64')

        if not grad:
            weightslice = calcGrid.calcASlice( posdata, weightdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
        else:
            graddata = self.data[grad][pp,:].astype('float64')
            weightslice = calcGrid.calcASlice( posdata, weightdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, graddata, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )

        if not grad:
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )
        else:
            graddata = self.data[grad][pp,:].astype('float64')
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, graddata, proj=proj, boxz=boxz, nz=nz, numthreads=numthreads )

        slice = data["grid"] / weightslice["grid"]

        if absolute:
            slice = np.abs( slice )

        if (not proj):
            neighbours = data[ "neighbours" ]
            contours = data[ "contours" ]
        x = pylab.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        y = pylab.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]

        if logplot:
            slice = pylab.maximum( slice, minimum )
            if dextoshow:
                 if not vrange:
                      if np.log10(slice.max() / slice.min()) > dextoshow:
                            vrange = [ slice.max() / 10.0**dextoshow, slice.max() ]
                 else:
                      print( "vrange already chosen so dextoshow is not applied" )

        if not logplot:
            print("dextoshow is not supported for non logarithmic plots")

        if not vrange:
            vrange = [ slice.min(), slice.max() ]

        print("Plot range", vrange)
        
        if cmap and not maskplot:
            pylab.set_cmap( cmap )


        if maskplot:
            pc = ax.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized, cmap=cmap )
        
        elif logplot:
            pc = ax.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized )
        
        else:
            if divzero: #added 30/05/23
                print('Divzero: ',vrange[0],vrange[1])
                pc = ax.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.TwoSlopeNorm(vmin=vrange[0], vmax=vrange[1], vcenter=divzero_centre), rasterized=rasterized)#, cmap=cmapref)
            else:
                pc = ax.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized, vmin=vrange[0], vmax=vrange[1])#, cmap=cmapref )
        
        #else:
        #    pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized, vmin=vrange[0], vmax=vrange[1] )
        '''
        if cmap:
            pylab.set_cmap( cmap )
        '''



        if colorbar:
            if logplot:
                cb = ax.colorbar( format=matplotlib.ticker.LogFormatterMathtext() )
            else:
                fmt = matplotlib.ticker.ScalarFormatter( useMathText=True )
                fmt.set_powerlimits( (-2, 2) )
                fmt.set_useOffset( False )
                cb = ax.colorbar( format=fmt )
            if cblabel:
                cb.set_label( cblabel )
        
        if contour:
            x = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[0] - .5 * box[0] + center[0]
            y = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[1] - .5 * box[1] + center[1]
            if not proj:
                    ax.contour( x, y, pylab.transpose( contours ), levels=[0.99], colors="w" )
            else:
                    ax.contour( x, y, pylab.transpose( slice ), levels=levels, colors="k" )

        #pylab.axis( "image" )

        if newlabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=24, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=24, ha='right' )
        return pc


    def plot_Aslice2( self, value, logplot=False, colorbar=False, res=1024, center=False, axes=[0,1], proj_fact=0.5, minimum=1e-8, newfig=True, newlabels=False, cmap=False, vrange=False, cblabel=False, rasterized=False, box=False, proj=True, cic=True, tsc=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = self.data['pos'].astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        #zdist = 2. * self.data['vol'].astype('float64')**(1./3.)
        if proj:
            zdist = proj_fact * box.max()

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )
        print("Selected %d of %d particles." % (pp.size,self.npart))

        posdata = pos[pp,:]
        valdata = self.data[value][pp].astype('float64')

        print("Total before projection is", valdata.sum())

        data = calcGrid.calcASlice2( posdata, valdata, res, res, box[0], box[1], c[0], c[1], axis0, axis1, proj=proj, cic=cic, tsc=tsc )
        slice = data["grid"]

        print("Total after projection is", slice.sum())

        x = pylab.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        y = pylab.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]

        if newfig:
            fig = pylab.figure()

        if logplot:
            #i,j = np.where(slice > 0.0)
            #minimum = slice[i,j].min()
            slice = pylab.maximum( slice, minimum )

        if not vrange:
            vrange = [ slice.min(), slice.max() ]
            print("Plot range ", vrange)
        
        if logplot:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized )
        else:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized, vmin=vrange[0], vmax=vrange[1] )
        
        if cmap:
            pylab.set_cmap( cmap )

        if colorbar:
            if logplot:
                cb = pylab.colorbar( format=matplotlib.ticker.LogFormatterMathtext() )
            else:
                fmt = matplotlib.ticker.ScalarFormatter( useMathText=True )
                fmt.set_powerlimits( (-2, 2) )
                fmt.set_useOffset( False )
                cb = pylab.colorbar( format=fmt )
            if cblabel:
                cb.set_label( cblabel )
        
        pylab.axis( "image" )

        if newlabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=24, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=24, ha='right' )
        return

    
    def plot_Aweightedslice2( self, value, weights, logplot=False, colorbar=False, res=1024, center=False, axes=[0,1], proj_fact=0.5, minimum=1e-8, newfig=True, newlabels=False, cmap=False, vrange=False, cblabel=False, rasterized=False, box=False, proj=False, cic=True, tsc=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = self.data['pos'].astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        zdist = 2. * self.data['vol'].astype('float64')**(1./3.)
        if proj:
            zdist = proj_fact * box.max()

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )
        print( "Selected %d of %d particles." % (pp.size,self.npart) )

        posdata = pos[pp,:]
        weightdata = self.data[weights][pp].astype('float64')
        valdata = self.data[value][pp].astype('float64') * weightdata

        weightslice = calcGrid.calcASlice2( posdata, weightdata, res, res, box[0], box[1], c[0], c[1], axis0, axis1, proj=proj, cic=cic, tsc=tsc )

        print( "Total before projection is", valdata.sum() )
        data = calcGrid.calcASlice2( posdata, valdata, res, res, box[0], box[1], c[0], c[1], axis0, axis1, proj=proj, cic=cic, tsc=tsc )
        print( "Total after projection is", data["grid"].sum() )

        slice = data["grid"]

        index1, index2 = np.where(weightslice["grid"] > 0.0)
        
        slice[index1, index2] /= weightslice["grid"][index1, index2]
        
        x = pylab.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        y = pylab.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]

        if newfig:
            fig = pylab.figure()

        if logplot:
            #i,j = np.where(slice > 0.0)
            #minimum = slice[i,j].min()
            slice = pylab.maximum( slice, minimum )

        if not vrange:
            vrange = [ slice.min(), slice.max() ]

        print( "Plot range ", vrange )
        
        if logplot:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized )
        else:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized, vmin=vrange[0], vmax=vrange[1] )
        
        if cmap:
            pylab.set_cmap( cmap )

        if colorbar:
            if logplot:
                cb = pylab.colorbar( format=matplotlib.ticker.LogFormatterMathtext() )
            else:
                fmt = matplotlib.ticker.ScalarFormatter( useMathText=True )
                fmt.set_powerlimits( (-2, 2) )
                fmt.set_useOffset( False )
                cb = pylab.colorbar( format=fmt )
            if cblabel:
                cb.set_label( cblabel )
        
        pylab.axis( "image" )

        if newlabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=24, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=24, ha='right' )
        return


    def reducesnap( self, idx, filename, verbose=False ):
        fout = open( filename, "w" )
        
        npart = pylab.zeros( 6, dtype=pylab.int32 )
        for ptype in range( 6 ):
            npart[ptype] = np.size( np.where( self.type[idx] == ptype ) )
        nparthighword = pylab.zeros( 6, dtype=pylab.int32 )
        
        la = pylab.zeros( 52, dtype=pylab.int8 )

        blksize = 256
        header = struct.pack( "i4sii", 8, 'HEAD', blksize+8, 8 )
        fout.write( header )
        
        fout.write( struct.pack( "i", blksize ) )
        npart.tofile( fout )
        self.masses.tofile( fout )
        fout.write( struct.pack( "ddii", self.time, self.redshift, self.flag_sfr, self.flag_feedback ) )
        npart.tofile( fout )
        fout.write( struct.pack( "iid", self.flag_cooling, 1, self.boxsize) )
        fout.write( struct.pack( "ddd", self.omega0, self.omegalambda, self.hubbleparam) )
        fout.write( struct.pack( "ii", self.flag_stellarage, self.flag_metals ) )
        nparthighword.tofile( fout )
        fout.write( struct.pack( "iii", self.flag_entropy_instead_u, self.flag_doubleprecision, self.flag_lpt_ics) )
        la.tofile( fout )
        fout.write( struct.pack( "i", blksize ) )
        
        needed = pylab.zeros( self.npartall, dtype='int32' )
        needed[idx] = 1
        
        for field in self.origdata:
            if field == "HEAD":
                continue
            key = field.lower().strip()
            
            count = pylab.zeros( 6 )
            hasfield = pylab.zeros( self.npartall, dtype='int32' )
            for ptype in range( 6 ):
                if self.parttype_has_block( ptype, field ):
                    count[ptype] = npart[ptype]
                    hasfield[ np.where(self.type == ptype) ] = 1
            tot = count.sum()
            
            if verbose:
                print("field:", field, "count:", count, "tot:", tot)
            
            if field in self.datablocks_int32:
                dtype = 'int32'
            else:
                dtype = 'float32'
            
            if tot > 0:
                ndim = self.data[key].ndim
                if ndim == 2:
                    dim = pylab.shape( self.data[key] )[1]
                    data = pylab.zeros( [tot,dim], dtype=dtype )
                else:
                    data = pylab.zeros( [tot], dtype=dtype )
            
                hidx, = np.where( (hasfield == 1) )
                pidx, = np.where( needed[hidx] == 1 )
                
                if verbose:
                    print("Field:", field, data.dtype, self.data[key].dtype)
                
                if ndim == 2:
                    data[:,:] = self.data[key][pidx,:]
                else:
                    data[:] = self.data[key][pidx]
            
                header = struct.pack( "i4sii", 8, field, data.size*4+8, 8 )
                fout.write( header )
            
                fout.write( struct.pack( "i", data.size*4 ) )
                data.tofile( fout )
                fout.write( struct.pack( "i", data.size*4 ) )
        fout.close()
        
    def load_species( self, speciesfile ):
        self.sp = loaders.load_species( speciesfile )
        return

    def get1DAbundanceMapping( self, maxradius, nshells, zmax, radioactives ):
        rad = self.pos[:self.nparticlesall[0],0].astype('f8')
        mass = self.data['mass'][:self.nparticlesall[0]].astype('f8')

        nradioactives = len( radioactives )
        radiolist = {}
        for iradio in range( nradioactives ):
            radiolist[ radioactives[iradio] ] = iradio

        dr = maxradius / nshells

        shell_mass, shell_edges = np.histogram( rad, bins=nshells, range=[0,maxradius], weights=mass )
        elements = np.zeros( (nshells,zmax+1) )
        radios = np.zeros( (nshells,nradioactives) )
        for iso in range( self.nspecies ):
            iz = self.sp['nz'][iso]
            if iz <= zmax:
                shell_elements, shell_edges =  np.histogram( rad, bins=nshells, range=[0,maxradius], weights=mass*self.data['xnuc'][:,iso] )
                elements[:,iz] += shell_elements

            if self.sp['names'][iso] in radiolist:
                radios[:,radiolist[ self.sp['names'][iso] ]] = shell_elements

        density = np.zeros( nshells )
        for ishell in range( nshells ):
            density[ishell] = shell_mass[ishell] / (4.0/3.0 * pi * dr**3 * ((ishell+1)**3 - ishell**3))

            if shell_mass[ishell] > 0:
                elements[ishell,:] /= shell_mass[ishell]
                radios[ishell,:] /= shell_mass[ishell]
            else:
                elements[ishell,:] = 0.
                radios[ishell,:] = 0.

        ige = elements[:,21:].sum(axis=1)

        return density, shell_mass, elements, radios, ige

    def get_most_bound_dm_particles(self, center=False):

        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        radius = pylab.zeros( self.npartall )
        for i in range( 3 ):
            radius += (self.data[ "pos" ][:,i] - center[i])**2
        radius = pylab.sqrt( radius )
        rs = radius.argsort()

        pid = self.data['id'][rs].astype('int64')
        ptype = self.type[rs]

        jj, = np.where( (ptype==1) )
        sidmallnow = pid[jj][:100]
        #jj, = np.where( (self.type == 1)  & (self.r() < 0.1) )
        #iddm = self.data['id'][jj].astype('int64')
        #sidmallnow = iddm[:100]

        return sidmallnow


    def force_halo_centering( self, sf, sidmallnow ):
        
        # Deal with offsets if gas present                             
        ndmp = self.nparticlesall[1]
        offsetdm = np.int_(self.nparticlesall[:1].sum())
        ndm = np.zeros(len(range(sf.data['fnsh'].sum())))
        nfof = -np.ones(len(range(sf.data['fnsh'].sum())))
        nsub = -np.ones(len(range(sf.data['fnsh'].sum())))
        subindarr = -np.ones(len(range(sf.data['fnsh'].sum())))
        offsetsub = 0

        #print('Looking for IDs:',sidmallnow)
        # Cross-check DM particles in each sub-halo with sidm ID list      
        #print('sf.data[fnsh]=',sf.data['fnsh'],sf.data['fnsh'].sum())
        #print('sf.data[slty][:, 1]=',sf.data['slty'][:, 1])
        #print('dm indices',self.nparticlesall[:1].sum(),self.nparticlesall[1])
        for i in range(len(sf.data['fnsh'])):
            offsetsub = sf.data['flty'][:i,1].sum()
            print('i,offsetsub=',i,offsetsub)
            for j in range(sf.data['fnsh'][i]):
                subind = sf.data['fnsh'][:i].sum() + j
                #print('subind=',subind)
                ndminhalo = sf.data['slty'][subind, 1].astype('int64')
                first_dmp = np.int_(offsetdm + offsetsub)
                last_dmp = np.int_(first_dmp + ndminhalo)
                offsetsub += ndminhalo
                #print('j,i,first_dmp,last_dmp=',j,i,first_dmp,last_dmp)
                idtmp = self.data['id'][first_dmp:last_dmp].astype('int64')
                idsort = np.argsort(idtmp)
                #print('idtmp=',idtmp[idsort])
                #sidmall = np.intersect1d(sidmallnow, idtmp, assume_unique=True)
                sidmall = np.where( np.in1d(sidmallnow, np.array(idtmp)) )
                #print('sidmall=',sidmall)
                #print('ravel sidmall=',np.ravel(sidmall))
                ndm[subind] = np.size(np.ravel(sidmall))
                if ndm[subind] > 0:
                    nfof[subind] = i
                    nsub[subind] = j
                    subindarr[subind] = subind

        print('ndm=',ndm)
        idm, = np.where( (self.data['type'] == 1) )
        ldmi = np.int_(offsetdm + self.nparticlesall[1])
        #print('offsetdm,ldmi=',offsetdm,ldmi)
        indy, = np.where( np.in1d(self.data['id'][offsetdm:ldmi], sidmallnow) )
        #print('indy=',indy)

        #print('ndm=',ndm)

        """
        # If most bound particles not in any sub-halo, look in fuzz        
        first_dmp = last_dmp.astype('int64')
        last_dmp = np.int_(ndmp + offsetdm)
        #print('first last for fuzz=',first_dmp,last_dmp)
        ntmp = last_dmp - first_dmp
        ilist = np.arange(ntmp)
        idtmp = self.data['id'][first_dmp:last_dmp].astype('int64')
        #sidmall = np.intersect1d(sidmallnow, idtmp, assume_unique=True)
        sidmall = np.where( np.in1d(sidmallnow, idtmp) )

        indices = np.arange(idtmp.shape[0])[np.in1d(idtmp, sidmallnow)]

        #print('sidmall=',sidmall)
        nfuz = len(sidmall)
        
        print('sub halo index with >0 dm particles=',np.where(ndm>0))
        print('nfuz=',nfuz)
        """
        # If fuzz, set centre to coordinates of most bound ID particle   
        if ( (np.max(ndm) == 0) ):
            nn = np.where( (idtmp == sidmall[0]) )
            nn += first_dmp
            center = [self.pos[nn[0][0],0], self.pos[nn[0][0],1], self.pos[nn[0][0],2]]
            jj = np.int_(sf.data['fnsh'].sum())
            ii = jj
            kk = jj
        # If in a sub-halo, assign centre to that sub-halo position      
        else:
            #jj = ndm.argmax()
            ind = ndm.argmax().astype('int64')
            jj = nsub[ind].astype('int64')
            ii = nfof[ind].astype('int64')
            kk = subindarr[ind].astype('int64')
            center = sf.data['spos'][kk,:]

        return kk,jj,ii,center


    # functions below are for use in combination with arepo_run
    def getTime(self):
        return self.time

    def computeMach(self, eos='ideal'):
        indgas = self.data['type'] == 0
        if eos=='ideal':
            gamma = 5./3.
            sound = np.sqrt(gamma * self.data['pres']/self.data['rho'])
        else:
            try:
                sound = np.zeros_like(self.data['rho'])
                for i, (r, t) in enumerate(zip(self.data['rho'], self.data['temp'])):
                    res = eos.tgiven(0.7, r, t)
                    sound[i] = sqrt(res['gamma1']*res['p']/r)
            except:
                gamma = 5./3.
                sound = np.sqrt(gamma * self.data['pres']/self.data['rho'])
        indgas = self.data['type'] == 0
        mach = np.sqrt((self.vel[indgas]**2).sum(axis=1)) / sound
        self.data['mach'] = mach
        self.data['sound'] = sound
        return

    def getMaxValue(self, value, eos='ideal'):
        if value in self.data:
            result = self.data[value].max()
        elif value == 'mach':
            self.computeMach(eos)
            result = self.data['mach'].max()
        else:
            raise Exception("Value %s does not exist."%(value))
        return result

    def getMeanValue(self, value, eos='ideal'):
        if value in self.data:
            result = self.data[value].mean()
        elif value == 'mach':
            self.computeMach(eos)
            result = self.data['mach'].mean()
        else:
            raise Exception("Value %s does not exist."%(value))
        return result

    def getAngularMomentum(self):
        rcm = self.centerofmass()
        return (self.mass.astype(np.float64)*( 
                (self.pos.astype(np.float64)[:,0]-rcm[0])
                  *self.vel.astype(np.float64)[:,1] - 
                (self.pos.astype(np.float64)[:,1]-rcm[1])
                  *self.vel.astype(np.float64)[:,0] )
            ).sum()

    def getAngularMomentumEnv(self):
        rcm = self.centerofmass()
        ind = self.data['type'] == 0
        return (self.mass[ind].astype(np.float64)*( 
                (self.pos[ind].astype(np.float64)[:,0]-rcm[0])
                  *self.vel[ind].astype(np.float64)[:,1] - 
                (self.pos[ind].astype(np.float64)[:,1]-rcm[1])
                  *self.vel[ind].astype(np.float64)[:,0] )
            ).sum()

    def getAngularMomentumCore(self):
        rcm = self.centerofmass()
        ind = self.data['type'] == 1
        return (self.mass[ind].astype(np.float64)*( 
                (self.pos[ind].astype(np.float64)[:,0]-rcm[0])
                  *self.vel[ind].astype(np.float64)[:,1] - 
                (self.pos[ind].astype(np.float64)[:,1]-rcm[1])
                  *self.vel[ind].astype(np.float64)[:,0] )
            ).sum()


    def getCenterOfMass(self):
        return np.array(self.centerofmass())

    def computeValueGas(self, value):
        """Compute value for the snapshot
        Returns 0 if successful, -1 otherwise
        Can be simply extended by adding more elif clauses
        """
        if value in self.data.keys():
            return 0
        indgas = self.data['type'] == 0
        if value == 'entr':
            self.data['entr'] = self.data['pres'] / self.data['rho']**(5./3.)
        elif value == 'vabs':
            self.data['vabs'] = np.sqrt((self.vel[indgas]**2).sum(axis=1))
        elif value == 'vr':
            inddm = self.data['type'] == 1
            cm = self.center()
            if inddm.sum() == 1:
                cm = self.pos[inddm].flatten()
            rad = self.pos[indgas] - cm[np.newaxis,:]
            self.data['vr'] = (self.vel[indgas] * rad).sum(axis=1) / \
                                np.sqrt((rad**2).sum(axis=1))
        elif value == 'mach':
            self.computeMach()
            #sound = np.sqrt(5./3. * self.data['pres']/self.data['rho'])
            #self.data['mach'] = np.sqrt((self.vel[indgas]**2).sum(axis=1))/sound
        elif value == 'temp':
            # assuming molecular weight of X=0.7, Y=0.3 -> 0.62
            R = 8.314e7
            cv = 1.5 * R/0.62
            self.data['temp'] = self.data['u']/cv
        elif value in ['E', 'x1h', 'x1he', 'x2he', 'uion', 'x0h', 'x0he']:
            self.computeIonizationState()
            self.data['x0h'] = 1. - self.data['x1h']
            self.data['x0he'] = 1. - self.data['x1he'] - self.data['x2he']
        elif value == 'mu':
            if not 'xnuc' in self.data:
                # assume solar composition
                self.data['xnuc'] = np.zeros((self.nparticlesall[0], 3))
                self.data['xnuc'][:,0] = 0.7
                self.data['xnuc'][:,1] = 0.28
                self.data['xnuc'][:,2] = 0.02
            # expect X, Y, Z in xnuc
            if not self.data['xnuc'].shape[1] == 3:
                print("For computing mu: unexpected shape, assuming that "
                      "first species is H and second species is He.")
            zi = np.array([1.0, 2.0, 6.0, 7.0, 8.0, 10.0])
            mui = np.array([1.0079, 4.0026, 12.011, 14.0067, 15.9994, 20.179])
            xi0 = np.array([0.0, 0.0, 0.247137766, 0.0620782, 0.52837118, 0.1624188])
            # mu of metals, to scale with Z
            muz = (xi0*mui).sum()
            xi = np.zeros((self.data['xnuc'].shape[0], 6))
            # set X and Y
            xi[:,0:2] = self.data['xnuc'][:,0:2]
            # set metal abundances
            for i in range(2, 6):
                xi[:,i] = xi0[i] * self.data['xnuc'][:,2] * mui[i] / muz
            self.data['mu'] = 1.0 / ((1.0 + zi[None,:]) * xi[:,:] / mui[None,:]).sum(axis=1)
        elif value == 'utherm':
            import const
            # constants
            sigma = 5.670400e-5
            a = sigma / const.c
            self.computeValueGas('mu')
            if not 'temp' in self.data:
                self.computeValueGas('temp')
            self.data['utherm'] = 1.5*self.data['temp']*const.KB/(self.data['mu']*const.amu) + a*self.data['temp']**4/self.data['rho']
        else:
            return -1
        return 0

    def computeIonizationState(self, verbose=False):
        """Compute ionization state of H/He mix"""
        # try to load from cache file
        if self.loadIonizationState():
            return
        # otherwise compute
        from ionization_hhe import IonizationState
        ion = IonizationState(X=0.7, Y=0.3)
        self.computeValueGas('temp')
        # create empty arrays
        self.data['E'] = np.zeros_like(self.data['pres'])
        self.data['x1h'] = np.zeros_like(self.data['pres'])
        self.data['x1he'] = np.zeros_like(self.data['pres'])
        self.data['x2he'] = np.zeros_like(self.data['pres'])
        self.data['uion'] = np.zeros_like(self.data['pres'])

        # loop over cells
        for i, (p, t) in enumerate(zip(self.data['pres'], self.data['temp'])):
            sol = ion(p, t)
            self.data['E'][i]    = sol[0]
            self.data['x1h'][i]  = sol[1]
            self.data['x1he'][i] = sol[2]
            self.data['x2he'][i] = sol[3]
            self.data['uion'][i] = sol[4]
            if verbose:
                if 10*i % self.data['pres'].shape[0] == 0:
                    print("Completed %d of %d -> %g %%"%(i, self.data['pres'].shape[0], 100*i/np.float(self.data['pres'].shape[0])))
        # save to cache
        self.saveIonizationState()

    def saveIonizationState(self):
        """Save ionization state of H/He mix"""
        import os, os.path
        # get directory from filename
        snapdir = os.path.dirname(self.files[0])
        # create cache dir
        cachedir = os.path.join(snapdir, 'ionizationdata')
        try:
            os.mkdir(cachedir)
        except OSError:
            pass
        # get snap number
        import re
        p = re.compile('\d\d\d')
        num = p.findall(self.files[0])[-1]
        cachefile = os.path.join(cachedir, 'ionization-snap%s.hdf5'%num)
        # save ionization data
        f = h5py.File(cachefile, "w")
        ndata = len(self.data['E'])
        for key in ['E', 'x1h', 'x1he', 'x2he', 'uion']:
            data = f.create_dataset("ionizationdata/"+key, (ndata,), dtype='d')
            data[...] = self.data[key]
        header = f.create_group('header')
        header.attrs['snapname'] = self.files[0]
        header.attrs['num'] = num
        header.attrs['time'] = self.time
        header.attrs['len'] = ndata
        f.close()

    def loadIonizationState(self):
        """Load ionization state of H/He mix, returns True when loaded, False else"""
        # already computed?
        if 'E' in self.data:
            return True
        import os, os.path
        # get directory from filename
        snapdir = os.path.dirname(self.files[0])
        # check for cache dir
        cachedir = os.path.join(snapdir, 'ionizationdata')
        if not os.path.exists(cachedir):
            return False
        # get snap number
        import re
        p = re.compile('\d\d\d')
        num = p.findall(self.files[0])[-1]
        cachefile = os.path.join(cachedir, 'ionization-snap%s.hdf5'%num)
        # load ionization data
        if not os.path.exists(cachefile):
            return False
        f = h5py.File(cachefile, "r")
        header = f['header']
        if self.time != header.attrs['time'] or len(self.data['rho']) != header.attrs['len']:
            print("Cache file found, but differs from snap!")
            return False
        for key in ['E', 'x1h', 'x1he', 'x2he', 'uion']:
            data = f["ionizationdata/"+key]
            self.data[key] = data[:]
        f.close()
        return True

    def getGwAmplitudes(self):
        result = {}
        cm = self.getCenterOfMass()
        result['Axx'] = (G/c**4 *(self.mass * (2.*self.vel[:,0]*self.vel[:,0] + (self.pos[:,0]-cm[0])*self.acc[:,0] + (self.pos[:,0]-cm[0])*self.acc[:,0]))).sum()
        result['Ayy'] = (G/c**4 *(self.mass * (2.*self.vel[:,1]*self.vel[:,1] + (self.pos[:,1]-cm[1])*self.acc[:,1] + (self.pos[:,1]-cm[1])*self.acc[:,1]))).sum()
        result['Azz'] = (G/c**4 *(self.mass * (2.*self.vel[:,2]*self.vel[:,2] + (self.pos[:,2]-cm[2])*self.acc[:,2] + (self.pos[:,2]-cm[2])*self.acc[:,2]))).sum()
        result['Axy'] = (G/c**4 *(self.mass * (2.*self.vel[:,0]*self.vel[:,1] + (self.pos[:,0]-cm[0])*self.acc[:,1] + (self.pos[:,1]-cm[1])*self.acc[:,0]))).sum()
        result['Axz'] = (G/c**4 *(self.mass * (2.*self.vel[:,0]*self.vel[:,2] + (self.pos[:,0]-cm[0])*self.acc[:,2] + (self.pos[:,2]-cm[2])*self.acc[:,0]))).sum()
        result['Ayz'] = (G/c**4 *(self.mass * (2.*self.vel[:,1]*self.vel[:,2] + (self.pos[:,1]-cm[1])*self.acc[:,2] + (self.pos[:,2]-cm[2])*self.acc[:,1]))).sum()

        result['Azplus'] = result['Axx'] - result['Ayy']
        result['Azcross'] = 2*result['Axy']

        result['Axplus'] = result['Azz'] - result['Ayy']
        result['Axcross'] = -2*result['Ayz']
        return dict2obj(result)

    def computeUnboundMassInternal(self):
        """compute unbound mass including internal energy"""
        indgas = self.data['type'] == 0
        energy = self.data['u'] + self.data['pot'][indgas] + 0.5 * ((self.data['vel'][indgas]**2).sum(axis=1))
        return self.data['mass'][indgas][energy > 0].sum()

    def computeUnboundMass(self):
        """compute unbound mass not including internal energy"""
        indgas = self.data['type'] == 0
        energy = self.data['pot'][indgas] + 0.5 * ((self.data['vel'][indgas]**2).sum(axis=1))
        return self.data['mass'][indgas][energy > 0].sum()

    def computeUnboundMassThermal(self):
        """compute unbound mass including thermal energy"""
        indgas = self.data['type'] == 0
        self.computeValueGas('utherm')
        energy = self.data['utherm'] + self.data['pot'][indgas] + 0.5 * ((self.data['vel'][indgas]**2).sum(axis=1))
        return self.data['mass'][indgas][energy > 0].sum()

    def getCenterOfMassPass0(self):
        return ((self.mass*self.data['pass00'])[:,np.newaxis]*self.pos).sum(axis=0)/(self.mass*self.data['pass00']).sum()

    def getCenterOfMassPass1(self):
        return ((self.mass*self.data['pass01'])[:,np.newaxis]*self.pos).sum(axis=0)/(self.mass*self.data['pass01']).sum()

    def computeMassin3a(self):
        """compute mass in volume with r=3a (a: orbit of binary)"""
        if self.nparticlesall[1] != 2:
            print("Computing mass in vol of r=3a works only for binary systems")
            return 0
        inddm = self.data['type'] == 1
        indgas = self.data['type'] == 0
        cmbinary = (self.pos[inddm]*self.mass[inddm][:,None]/self.mass[inddm].sum()).sum(axis=0)
        distance = np.sqrt(((self.pos[inddm][0] - self.pos[inddm][1])**2).sum())
        indvol = np.sqrt(((self.pos[indgas] - cmbinary[None,:])**2).sum(axis=1)) < 3.*distance
        return self.mass[indgas][indvol].sum()


    def read_starparticle_mergertree_data(self, snap, path, name):
        blocknames = {'Insitu':['ParticleIDs'], 'Exsitu':['AccretedFlag', 'BirthFoFindex', 'BirthSubhaloindex', 'ParticleIDs', 'PeakMassIndex', 'RootIndex', 'BoundFirstTime', 'PeakMassInfalltime']}
        groupnames = ['Insitu', 'Exsitu']
        
        nins = 0
        nacc = 0
        self.mdata = {}

        datayes = np.zeros(snap+1)
        for i in range(snap+1):
            afname = path + '/%sstarID_accreted_all_newmtree_%03d.hdf5'%(name,i)
            if os.path.exists(afname):
                ff = h5py.File(afname, 'r')
                try:
                    header = ff['/Header'].attrs
                    if 'n_ins' in header:
                        nins += header['n_ins']
                        datayes[i] = 1
                    if 'n_acc' in header:
                        nacc += header['n_acc']
                        datayes[i] = 1
                except:
                    print('Warning: no data for file %s',afname)

                
        self.mdata['Insitu'] = {}
        self.mdata['Exsitu'] = {}
        self.mdata['Insitu']['Npart'] = nins
        self.mdata['Exsitu']['Npart'] = nacc
        print('%d in-situ and %d ex-situ stars found'%(nins,nacc))
            
        for groupname in groupnames:
            for block in blocknames[groupname]:
                if block not in ff[groupname]:
                    continue
                for i in range(snap+1): # find a file that has both Exsitu and Insitu data to get datatypes...
                    afname = path + '/%sstarID_accreted_all_newmtree_%03d.hdf5'%(name,i)
                    if os.path.exists(afname):
                        ff = h5py.File(afname, 'r')
                    try:
                        expres = ff['Exsitu']
                        inpres = ff['Insitu']
                        break
                    except:
                        pass
                         
                datatype = ff[groupname][block].dtype
                self.mdata[groupname][block] = np.empty(self.mdata[groupname]['Npart'], dtype=datatype)
                
        self.mdata['Exsitu']['BirthSnap'] = np.empty(self.mdata[groupname]['Npart'], dtype='i4')                
                
        firstins = 0
        firstacc = 0
        lastins = 0
        lastacc = 0
        indices, = np.where(datayes==1)
        for i in indices:            
            afname = path + '/%sstarID_accreted_all_newmtree_%03d.hdf5'%(name,i)
            if os.path.exists(afname):
                ff = h5py.File(afname, 'r')
                header = ff['/Header'].attrs
                nins = header['n_ins']
                nacc = header['n_acc']
                lastins += nins
                lastacc += nacc
                groupname = 'Insitu'
                if groupname in ff.keys():
                    for block in blocknames[groupname]:
                        if block not in ff[groupname]:
                            continue
                        self.mdata[groupname][block][firstins:lastins] = ff[groupname][block]

                groupname = 'Exsitu'
                if groupname in ff.keys():
                    for block in blocknames[groupname]:
                        if block not in ff[groupname]:
                            continue
                        self.mdata[groupname][block][firstacc:lastacc] = ff[groupname][block]
                    self.mdata[groupname]['BirthSnap'][firstacc:lastacc] = np.array([i]*nacc)
                firstins = lastins
                firstacc = lastacc
