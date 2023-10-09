import os
import numpy as np
import scipy
import glob
import rsr
import matplotlib.pyplot as plt
import h5py
import hdf5storage
from .. import filtering, surface


class Results:
    """ Class to read results as generated by the Trento simulator [Gerekos et al., 2018]
    """
    def __init__(self, 
                 simulation_name: str, # Folder name of the simulation
                 source_folder=('..', 'data', 'RadarSimulationResults')):
        
        self.simulation_name = simulation_name
        self.output_folder = os.path.join(*source_folder, simulation_name, 'OUTPUTS')
        self.input_folder = os.path.join(*source_folder, simulation_name, 'INPUTS')
        self.input_filenames = glob.glob(os.path.join(self.input_folder, '*.mat'))
        self.trajectory = self.read_trajectory()
        self.simulation = self.read_simulation()
        self.geoelectric = self.read_geoelectric()
    
    
    def frame_filenames(self):
        """ Provide the list of frame filenames 
        """
        out = glob.glob(os.path.join(self.output_folder, '*.mat'))
        out = np.sort(out)
        return out
    
    
    def loadmat(self, filename):
        """Load matfile
        """
        mat = hdf5storage.loadmat(filename)
        out = dict()
        for key in mat.keys():
            data = mat[key][0]
            if np.shape(data)[0] == 1:
                data = data[0]
            out[key] = data
        return out
    
            
    def read_simulation(self):
        filename = [filename for filename in self.input_filenames if 'Sim' in filename][0]
        out = self.loadmat(filename)
        return out
    
    
    def read_geoelectric(self):
        filename = [filename for filename in self.input_filenames if 'Geo' in filename][0]
        out = self.loadmat(filename)
        
        # Add the DEM (for any reason, only h5py sees it, not hdf5storage)
        f = h5py.File(filename)
        for key in f.keys() :
            if 'group' in f[key].__str__():
                for dset in f[key].keys():  
                    ds_data = f[key][dset] # returns HDF5 dataset object
                    arr = f[key][dset][:] # adding [:] returns a numpy array
                    out[dset] = arr
        
        return out
    
    
    def read_trajectory(self):
        filename = [filename for filename in self.input_filenames if 'Traj' in filename][0]
        out = self.loadmat(filename)
        return out
    
    
    def dem(self):
        """Provide the input DEM
        """
        out = self.geoelectric['b']
        out = np.rot90(out, k=1)
        return out
    
    
    def xy(self, meters=True):
        """Provide [x, y] coordinates of the frames
        """
        #x0_meter = self.input_data('Traj')['sc_position_x']
        #y0_meter = self.input_data('Traj')['sc_position_y']
        dx = self.geoelectric['DELTA_X']
        #length = int(self.input_data('Traj')['TrackLengthTot'])
        
        xs = self.trajectory['sc_position_x']
        ys = self.trajectory['sc_position_y']
        
        #xs = np.full(length, x0_meter)#np.arange(length)*dx + x0_meter
        #ys = np.arange(length)*dx + y0_meter
        
        if not meters:
            # Values in pixels
            xs = xs/dx
            ys = ys/dx
        
        return xs, ys
    
    
    def plot_trajectory(self):
        # Parameters
        dem = self.dem()
        dx = self.geoelectric['DELTA_X']
        x, y = self.xy(meters=True)
        radius = self.simulation['PulseLtdR']
        x_length = self.geoelectric['l']*dx
        y_length = self.geoelectric['m']*dx

        # FIGURE
        # ------
        fig, ax = plt.subplots()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(self.simulation_name)
        
        # DEM plot
        #im = ax.imshow(dem, extent=[0,x_length*dx,0,y_length*dx])
        im = ax.imshow(dem, extent=[y_length,0,x_length,0])

        # colorbar
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(im, cax=cax, label='Elevation [m]') # Similar to fig.colorbar(im, cax = cax)
        
        # Trajectory plot
        ax.plot(x, y, 'k',)

        # Footprint plot
        for i in [0, -1]:
            circle = plt.Circle((x[i],y[i]), radius, fill = False , color='k', ls='--')
            ax.set_aspect( 1 )
            ax.add_artist( circle )
            
        return ax
        

    def plot_radargram(self, compression='Hann windowing', aspect=6, pdb=True):
        """Browse product of the results
        """
        fast_time = self.simulation['t']
        
        fig, ax = plt.subplots()
        
        rdg = self.radargram(compression=compression, absolute=True, pdb=pdb, rotate=True)
        
        im = ax.imshow(rdg, extent=[0, np.shape(rdg)[1], 
                                    fast_time[-1]*1e6, fast_time[0]*1e6], aspect=aspect)
        
        # colorbar
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(im, cax=cax, label='Power [dB]') # Similar to fig.colorbar(im, cax = cax)
        
        ax.set_title(self.simulation_name)
        ax.set_ylabel(r'Time [$\mu$s]')
        ax.set_xlabel(r'Range bin')
        
        return ax
    
    
    def read_frame(self, i, compression=False):
        """Read a frame (range line)
        i = frame number
        """
        i = i-1
        out = scipy.io.loadmat(self.frame_filenames()[i])['Final_signal'][0]
        
        if compression == 'Hann windowing':
            out = filtering.pulse_compression(self.simulation['Signal'], out)
        elif compression == 'No windowing':
            out = filtering.pulse_compression(self.simulation['Signal_raw'], out)
        
        return out
    
    
    def plot_frame(self, i):
        """plot a frame
        i = frame number
        """
        # Signal
        raw_signal = self.read_frame(i)
        nowin_signal = self.read_frame(i, compression='No windowing')
        win_signal = self.read_frame(i, compression='Hann windowing')
        time = self.simulation['t']*1e6

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(15, 4))

        ax = axs[0]
        # Plot the measured signal in time domain
        ax.plot(time, np.absolute(raw_signal), lw=.5, label='Raw')
        ax.set_title('Measured Signal in Time Domain')
        ax.set_xlabel(r'Time [$\mu$s]')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

        ax = axs[1]
        # Plot the compressed signal in time domain
        ax.plot(time, np.absolute(win_signal), label='No windowing')
        # Plot the compressed signal in time domain
        ax.plot(time, np.absolute(win_signal), label='Hann windowing')
        ax.set_title('Compressed Signal in Time Domain')
        ax.set_xlabel(r'Time [$\mu$s]')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend()
        
        return axs
        
        
    
    def radargram(self, 
                  rotate=False,       # Rotate array for convenient display
                  absolute=False,     # Provide the absolute values instead of complex number
                  pdb=False,         # Provide power in dB
                  **kwargs,          # Anything to pass to read_frame (including compression options)
                 ):
        """ Stack the range lines to produce a 2D radargram
        """
        Nx = len(self.frame_filenames())
        Ny = len(self.simulation['signal_window'])
        
        out = np.empty((Nx, Ny),dtype=np.complex_)
        for i in np.arange(Nx):
            frame = self.read_frame(i, **kwargs)
            out[i,:] = frame
        
        if rotate:
            out = np.rot90(out)
        
        if absolute:
            out = np.absolute(out)
            
        if pdb:
            out = 20*np.log10(out)
            
        return out
        
        
    def surface(self, method='grima2012', compression='No windowing'):
        """Surface echo extraction
        """
        rdg = self.radargram(compression=compression)
        ys = surface.detector(rdg, axis=0, method=method)
        amps = [rdg[i, int(y)] for i, y in enumerate(ys)]
        return {'y':ys, 'amp':amps}
        
        
    def surface_rsr(self, method='grima2012', fit_model='hk'):
        """RSR
        """
        srf = self.surface(method=method)
        amp = np.absolute(srf['amp'])
        f = rsr.run.processor(amp, fit_model=fit_model)
        return f