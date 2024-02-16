import os
import time
from datetime import datetime
import configparser
import math, numpy as np

import xarray as xr, netCDF4, h5netcdf
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import MinuteLocator, DateFormatter
import matplotlib.colors as colors

# import pytmatrix

class FlightTrack:
    def __init__(self, config_fname):
        config = configparser.ConfigParser()
        config.read(config_fname)

        # load datasets
        if config.has_option('DEFAULT', 'hvps_fname'):
            microphysics_dataset_HVPS = xr.open_dataset(config['DEFAULT']['hvps_fname'], group='D_s')
        if config.has_option('DEFAULT', '2ds_fname'):
            microphysics_dataset_2DS = xr.open_dataset(config['DEFAULT']['2ds_fname'], group='D_s')
        if config.has_option('DEFAULT', 'bulk_fname'):
            bulk_dataset = xr.open_dataset(config['DEFAULT']['bulk_fname'])
        if config.has_option('DEFAULT', 'atm_state_fname'):
            atm_state = xr.open_dataset(config['DEFAULT']['atm_state_fname'])
        # if config.has_option('DEFAULT', 'radar_folder'):
        #     radar_folder = config['DEFAULT']['radar_folder']
        if config.has_option('DEFAULT', 'radar_fname'):
            radar_fname = config['DEFAULT']['radar_fname']

        # hvps vars
        if config.has_option('DEFAULT', 'hvps_fname'):
            self.time = microphysics_dataset_HVPS['time'] # time used for all other datasets
            self.size_hvps = microphysics_dataset_HVPS['size_dist_all_in'].T.data
            self.conc_hvps = microphysics_dataset_HVPS['size_dist_all_in'].T.data
            self.bin_mids_hvps = microphysics_dataset_HVPS['bin_mids']

        # 2ds vars
        if config.has_option('DEFAULT', '2ds_fname'):
            self.size_2ds = microphysics_dataset_2DS['size_dist_all_in'].T.data
            self.conc_2ds = microphysics_dataset_2DS['size_dist_all_in'].T.data
            self.bin_mids_2ds = microphysics_dataset_2DS['bin_mids']

        # combine hvps and 2ds
        # if config.has_option('DEFAULT', '2ds_fname') and config.has_option('DEFAULT', 'hvps_fname'):
        #     self.combine_2ds_hvps()

        # twc and lwc vars
        if config.has_option('DEFAULT', 'bulk_fname'):
            self.twc = bulk_dataset['NevTWC'].interp(time=self.time)
            self.lwc = (bulk_dataset['NevLWC']).interp(time=self.time)

        # atm state vars
        if config.has_option('DEFAULT', 'atm_state_fname'):
            self.Ts = atm_state['Ts'].interp(time=self.time)
            self.Ps = atm_state['Ps'].interp(time=self.time)

        # radar vars
        # # loop through radar files
        # if config.has_option('DEFAULT', 'radar_folder') and config.has_option('DEFAULT', 'radar_folder'):
        #     radar_fnames = [f for f in os.listdir(radar_folder) if f.endswith('.nc')]
        #     self.Z_ls = []
        #     self.Z_times = []
        #     for radar_fname in radar_fnames:
        #         radar_dataset = xr.open_dataset(os.path.join(radar_folder, radar_fname)) # open in readonly mode

        #         radar_time = radar_dataset['time']
        #         Z = radar_dataset['reflectivity']
        #         Z['profile'] = radar_dataset['time']
        #         Z = Z.rename({'profile':'time'})

        #         # reflectivity interpolation
        #         start_idx = np.searchsorted(self.time.data, radar_time[0], side="left")-1
        #         end_idx = np.searchsorted(self.time.data, radar_time[-1], side="left")-1
        #         time_trunc = self.time.isel(time=slice(start_idx,end_idx))
        #         self.Z_ls.append(Z.interp(time=time_trunc))

        #         self.Z_times.append((Z.time[0].data, Z.time[-1].data))
        if config.has_option('DEFAULT', 'radar_fname'):
            radar_dataset = xr.open_dataset(radar_fname)

            radar_time = radar_dataset['time']
            self.Z = radar_dataset['reflectivity']
            self.Z['profile'] = radar_dataset['time']
            self.Z = self.Z.rename({'profile':'time'})

            # reflectivity interpolation
            start_idx = np.searchsorted(self.time.data, radar_time[0], side="left")-1
            end_idx = np.searchsorted(self.time.data, radar_time[-1], side="left")-1
            time_trunc = self.time.isel(time=slice(start_idx,end_idx))
            self.Z.interp(time=time_trunc)

    # def combine_2ds_hvps(self):

    # filtering
    def filter(self, time_bounds=(np.datetime64(datetime.min),np.datetime64(datetime.max)), twc_bounds=(-1*np.inf,np.inf), range_bounds=(0,300)):
        # datasets to filter
        dataset_ls = []
        if hasattr(self, 'microphysics_dataset_HVPS'):
            dataset_ls.append( 'microphysics_dataset_HVPS')
        if hasattr(self, 'microphysics_dataset_2DS'):
            dataset_ls.append('microphysics_dataset_2DS')
        if hasattr(self, 'bulk_dataset'):
            dataset_ls.append('bulk_dataset')
        if hasattr(self, 'atm_state'):
            dataset_ls.append('atm_state')
        if hasattr(self, 'Z'):
            dataset_ls.append('Z')

        # filtering condition
        cond = (self.time > time_bounds[0], self.time < time_bounds[1], self.twc > twc_bounds[0], 
                self.twc < twc_bounds[1], self.Z.range > range_bounds[0], 
                self.Z.range < range_bounds[2])

        for dataset in dataset_ls:
            setattr(self, getattr(self, dataset).where(cond))

    # PLOTTING
    def plot_Ts(self, title, saveas=None):
        # temperature
        plt.figure(figsize=(10,5))

        plt.plot(self.time, self.Ts)

        plt.title(title, fontsize=20)
        plt.xlabel('Time [hh:mm]', fontsize=15)
        plt.ylabel('Ts [C]', fontsize=15)

        ax = plt.gca(); fig = plt.gcf()
        ax.xaxis.set_major_locator(MinuteLocator(byminute=[0,30]))
        xformatter = DateFormatter('%H:%M')
        fig.axes[0].xaxis.set_major_formatter(xformatter)

        if type(saveas) != type(None):
            plt.savefig(saveas)

    def plot_Ps(self, title, saveas=None):
        # pressure
        plt.figure(figsize=(10,5))

        plt.plot(self.time, self.Ps)

        plt.title(title, fontsize=20)
        plt.xlabel('Time [hh:mm]', fontsize=15)
        plt.ylabel('Ps [mB]', fontsize=15)

        ax = plt.gca(); fig = plt.gcf()
        ax.xaxis.set_major_locator(MinuteLocator(byminute=[0,30]))
        xformatter = DateFormatter('%H:%M')
        fig.axes[0].xaxis.set_major_formatter(xformatter)

        if type(saveas) != type(None):
            plt.savefig(saveas)

    def plot_TWC(self, title, saveas=None):
        plt.figure(figsize=(10,5))

        plt.scatter(self.time, self.twc, s=0.1, label='TWC')
        plt.scatter(self.time, self.lwc, s=0.1, label='LWC')
        plt.title(title, fontsize=20)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('[gm^-3]', fontsize=15)

        ax = plt.gca(); fig = plt.gcf()
        ax.xaxis.set_major_locator(MinuteLocator(byminute=[0,30]))
        xformatter = DateFormatter('%H:%M')
        fig.axes[0].xaxis.set_major_formatter(xformatter)

        plt.legend()

        if type(saveas) != type(None):
            plt.savefig(saveas)

    def plot_Z(self, title, saveas):
        plt.figure(figsize=(10,5))

        # plt.pcolormesh(self.Z_ls[i].time, self.Z_ls[i].range[:12], self.Z_ls[i].sel(np=0,range=slice(0,500)).T, 
        #             norm=colors.LogNorm())
        plt.pcolormesh(self.Z.time, self.Z.range[:12], self.Z.sel(np=0,range=slice(0,500)).T, 
                    norm=colors.LogNorm())

        plt.title(title, fontsize=20)
        plt.xlabel('Time [hh:mm]', fontsize=15)
        plt.ylabel('Range [m]', fontsize=15)

        ax = plt.gca(); fig = plt.gcf()

        if (self.Z.time[-1] - self.Z.time[0]).data.astype('timedelta64[s]') > 60*15:
            minute_arr = np.arange(0, 60, 5)
        else:
            minute_arr = np.arange(0, 60, 1)
        ax.xaxis.set_major_locator(MinuteLocator(byminute=minute_arr))
        xformatter = DateFormatter('%H:%M')
        fig.axes[0].xaxis.set_major_formatter(xformatter)
        ax.set_ylim([200,500])

        cbar = plt.colorbar()
        cbar.set_label('Reflectivity [mm^6/m^3]', fontsize=15)

        if type(saveas) != type(None):
            plt.savefig(saveas)

    def plot_PSD_ts(self, title, saveas, oap='hvps'):
        oap = ['hvps', '2ds', 'combined']
        if oap == 'hvps':
            size_dist = self.size_hvps
        elif oap == '2ds':
            size_dist = self.size_2ds
        elif oap == 'combined':
            size_dist = self.size_combined
        else:
            raise ValueError(f"Invalid OAP. Expected one of: {oap}")

        plt.figure(figsize=(10,5))
        ax = plt.gca(); fig = plt.gcf()

        plt.pcolormesh(self.time, self.bin_mids, size_dist+1,
                        norm=colors.LogNorm())
        plt.title(title, fontsize=20)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Bins [um]', fontsize=15)

        ax.xaxis.set_major_locator(MinuteLocator(byminute=[0,30]))
        xformatter = DateFormatter('%H:%M')
        fig.axes[0].xaxis.set_major_formatter(xformatter)
        # plt.yscale('log')

        cbar = plt.colorbar()
        cbar.set_label('Counts', fontsize=15)

        if type(saveas) != type(None):
            plt.savefig(saveas)

    def plot_conc_mean(self, saveas, oap='hvps'):
        oap_ls = ['hvps', '2ds', 'combined']
        if oap == 'hvps':
            conc_dist = self.conc_hvps
            bin_mids = self.bin_mids_hvps
        elif oap == '2ds':
            conc_dist = self.conc_2ds
            bin_mids = self.bin_mids_2ds
        elif oap == 'combined':
            conc_dist = self.conc_combined
            bin_mids = self.bin_mids_combined
        else:
            raise ValueError(f"Invalid OAP. Expected one of: {oap_ls}")

        plt.figure(figsize=(10,5))
        ax = plt.gca(); fig = plt.gcf()

        plt.step(self.bin_mids_hvps, np.mean(self.conc_hvps, axis=1), label='HVPS')
        plt.step(self.bin_mids_2ds, np.mean(self.conc_2ds, axis=1), label='2DS')
        plt.title('F01: Concentration Averaged Over Time')
        plt.ylabel('Average Concentration [#/L/um]', fontsize=15)
        plt.xlabel('Bins [um]', fontsize=15)
        plt.yscale('log')
        plt.xscale('log')

        plt.legend()

        if type(saveas) != type(None):
            plt.savefig(saveas)

    # def get_SV(self, start_idx, end_idx):
    #     V = 0.4 * (self.time[start_idx] = self.time[end_idx]) # assuming plane travelling at 100 m/s, sample volume is 400 L/s
    #     return np.insert(V, 0, V[0])

    def get_TWC(a, b, SV_inv, bin_mids, psd):
        TWC = np.zeros((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                TWC[i,j] = np.sum(a[i] * (bin_mids*1e-4)**b[j] * np.dot(psd, SV_inv))

        return TWC

    def get_SV(time_bins):
        V = 0.31 * (time_bins[1:] - time_bins[:1]) # assuming plane travelling at 100 m/s, sample volume is 310 L/s
        return np.insert(V, 0, V[0])

    def get_Z(a, b, SV_inv, bin_mids, psd):
        # a in gm/cm^b, b unitless
        # bins in um
        # psd is 2d array; 1st dim is diameter (m), 2nd dim is time, 

        Z = np.zeros((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                Z[i,j] = (6*0.17)/(math.pi*0.91*0.93) * np.sum((a[i] * (bin_mids*1e-4)**b[j])**2 * np.dot(psd, SV_inv))

        return Z

if __name__ == '__main__':
    config_fname = 'config/config_f01.ini'

    # plot atm state, lwc/twc, and number count
    # for i in range(1,10):
    #     # no hvps data for f04
    #     if i == 4:
    #         continue

    #     flighttrack = FlightTrack(f'config/config_f0{i}.ini')
    #     flighttrack.plot_Ts(f'F0{i} Temperature ', fr'D:\thesis\figs\f0{i}\temperature.png')
    #     flighttrack.plot_Ps(f'F0{i} Pressure', fr'D:\thesis\figs\f0{i}\pressure.png')
    #     flighttrack.plot_TWC(f'F0{i} Total/Liquid Water Content', fr'D:\thesis\figs\f0{i}\twc.png')
    #     flighttrack.plot_PSD(f'F0{i} HVPS Number Count', fr'D:\thesis\figs\f0{i}\hvps_psd.png')

    config_fname = 'config/config_f01.ini'
    start = time.time()
    f01 = FlightTrack(config_fname)
    print(time.time()-start)

    f01.plot_Z('F01 Reflectivity', 'z.png')
    # f01.plot_conc_mean('conc_mean.png', 'hvps')

    # for i in range(len(f01.Z_ls)):
    #     start_time = pd.to_datetime(f01.Z_ls[i].time.data[0])
    #     import matplotlib
    #     matplotlib.use('Agg')

    #     saveas = fr'D:\thesis\figs\f01\Z\Z_{start_time.strftime("%Y%m%dT%H%M%S")}.png'
    #     title = f'F01 Reflectivity' #({start_time.strftime("%Y-%m-%d %H:%M:%S")})'
    #     f01.plot_Z(title, i, saveas)