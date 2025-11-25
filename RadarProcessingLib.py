# Radar signal processing 
# based on rpl.py from RADIal

import os
import numpy as np
import mkl_fft
from scipy import signal
import math
import pandas as pd


class CA_CFAR():
    """
    Description:
    ------------
        Cell Averaging - Constant False Alarm Rate algorithm
        Performs an automatic detection on the input range-Doppler matrix with an adaptive thresholding.
        The threshold level is determined for each cell in the range-Doppler map with the estimation
        of the power level of its surrounding noise. The average power of the noise is estimated on a
        rectangular window, that is defined around the CUT (Cell Under Test). In order the mitigate the effect
        of the target reflection energy spreading some cells are left out from the calculation in the immediate
        vicinity of the CUT. These cells are the guard cells.
        The size of the estimation window and guard window can be set with the win_param parameter.
    Implementation notes:
    ---------------------
        Implementation based on https://github.com/petotamas/APRiL
    Parameters:
    -----------
    :param win_param: Parameters of the noise power estimation window
                      [Est. window length, Est. window width, Guard window length, Guard window width]
    :param threshold: Threshold level above the estimated average noise power
    :type win_param: python list with 4 elements
    :type threshold: float
    Return values:
    --------------
    """

    def __init__(self, win_param, threshold, rd_size):
        win_width = win_param[0]
        win_height = win_param[1]
        guard_width = win_param[2]
        guard_height = win_param[3]

        # Create window mask with guard cells
        self.mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
        self.mask[win_height - guard_height:win_height + 1 + guard_height, win_width - guard_width:win_width + 1 + guard_width] = 0

        # Convert threshold value
        self.threshold = 10 ** (threshold / 10)

        # Number cells within window around CUT; used for averaging operation.
        self.num_valid_cells_in_window = signal.convolve2d(np.ones(rd_size, dtype=float), self.mask, mode='same')

    def __call__(self, rd_matrix):
        """
        Description:
        ------------
            Performs the automatic detection on the input range-Doppler matrix.
        Implementation notes:
        ---------------------
        Parameters:
        -----------
        :param rd_matrix: Range-Doppler map on which the automatic detection should be performed
        :type rd_matrix: R x D complex numpy array
        Return values:
        --------------
        :return hit_matrix: Calculated hit matrix
        """
        # Convert range-Doppler map values to power
        rd_matrix = np.abs(rd_matrix) ** 2

        # Perform detection
        rd_windowed_sum = signal.convolve2d(rd_matrix, self.mask, mode='same')
        rd_avg_noise_power = rd_windowed_sum / self.num_valid_cells_in_window
        rd_snr = rd_matrix / rd_avg_noise_power
        hit_matrix = rd_snr > self.threshold

        return hit_matrix

class CASO_CFAR():
    """
    Description:
    ------------
        Cell-Averaging Smallest Of-CFAR (CASO-CFAR) 

    Implementation notes:
    ---------------------
        Implementation based on TI Automotive Toolbox 3.6.0 Lab0002: short range radar

    Parameters:
    -----------
    :param win_param: Parameters of the noise power estimation window
                      [Est. window length, Est. window width, Guard window length, Guard window width]
    :param threshold: Threshold level above the estimated average noise power
    :type win_param: python list with 4 elements
    :type threshold: float
    Return values:
    --------------
    """        
    
    def __init__(self, radar_params, CFAR_params):
        # Parameters in TI short range radar demo
        # cfarCfgDoppler.winLen = 8
        # cfarCfgDoppler.guardLen = 4
        # cfarCfgDoppler.noiseDivShift = 4  #Should be log2(2*winLen), used to implement dividing by right shift operation
        # cfarCfgDoppler.cyclicMode = 0

        # https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/1123346/iwr6843aopevm-cfar-ca-or-cfar-caso
        # SUBFRAME_SRR_MIN_SNR_dB = 15  # AWR6843, When Tx1 and Tx3 was used, minimum value of SNR was 15.

        # Fractional bit width for Thresholds for CFAR data
        # CFARTHRESHOLD_N_BIT_FRAC = 8 (bits)
        
        self.winLen = CFAR_params["winLen"]
        self.guardLen = CFAR_params["guardLen"]
        self.SUBFRAME_SRR_MIN_SNR_dB = CFAR_params["SUBFRAME_SRR_MIN_SNR_dB"]
        self.CFARTHRESHOLD_N_BIT_FRAC = CFAR_params["CFARTHRESHOLD_N_BIT_FRAC"]

        # self.thresholdScale = self.convertSNRdBtoThreshold(radar_params["num_Rx"]*radar_params["num_Tx"], \
        #                                         self.SUBFRAME_SRR_MIN_SNR_dB, self.CFARTHRESHOLD_N_BIT_FRAC)

    # -----------------------------------------------------------------------------------------------------------------
    # Converts an SNR (in dB) to an SNR Threshold that the CFAR algo can use
    # def convertSNRdBtoThreshold(self, numInteg, ThresholdIndB, bitwidth):
    #     thresholdScale = ((1 << bitwidth) * numInteg) * ThresholdIndB / 6.0
    #     return thresholdScale
        

    # -----------------------------------------------------------------------------------------------------------------    
    # CFAR in Doppler direction, wrap padding the data
    #
    # TI: Performs a CFAR on an 16-bit unsigned input vector (CFAR-CA). The input values are assumed to be
    # in lograthimic scale. So the comparision between the CUT and the noise samples is additive
    # rather than multiplicative. Comparison is two-sided (wrap around when needed) for all CUTs.
    # Input: 
    #       RD_dB: range-doppler spectrum in dB, 256x64
    def CASO_dBwrap_withSNR(self, RD_dB):
        guardLen    = self.guardLen
        noiseLen    = self.winLen

        RD = np.pad(RD_dB, ((0,0),(guardLen+noiseLen, guardLen+noiseLen)), 'wrap')
        sumLeft     = np.zeros((RD_dB.shape))
        sumRight    = np.zeros((RD_dB.shape))
        for idx in range(RD_dB.shape[1]):   # for every doppler bin
            idxPad = idx + guardLen + noiseLen
            sumLeft[:,idx]  = np.sum(RD[:,idxPad-guardLen-noiseLen:idxPad-guardLen], axis=1)
            sumRight[:,idx] = np.sum(RD[:,idxPad+guardLen:idxPad+guardLen+noiseLen], axis=1)
        sumLeft  = sumLeft.reshape(sumLeft.shape[0], sumLeft.shape[1], -1)
        sumRight = sumRight.reshape(sumRight.shape[0], sumRight.shape[1], -1)
        sumLR = np.concatenate((sumLeft,sumRight), axis=2)
        sumMin = np.squeeze(np.min(sumLR, axis=2))
        noise = sumMin/noiseLen

        det_points = (RD_dB > noise + self.SUBFRAME_SRR_MIN_SNR_dB).astype(int)
        doppler_SNR_dB = RD_dB - noise
        return det_points, doppler_SNR_dB

    # -----------------------------------------------------------------------------------------------------------------
    # CFAR in Range direction, no padding
    #
    # TI: Performs a CFAR SO on an 16-bit unsigned input vector. The input values are assumed to be
    # in lograthimic scale. So the comparision between the CUT and the noise samples is additive
    # rather than multiplicative.
    # Input: 
    #       RD_dB: range-doppler spectrum in dB, 256x64
    #       minIndxToIgnoreHPF: the number of indices to force one sided CFAR, so as to avoid false detections due to effect of the HPF.
    def CASO_dB_withSNR(self, RD_dB):
        guardLen    = self.guardLen
        noiseLen    = self.winLen

        sumTop = np.zeros((RD_dB.shape))   # he top noise cells
        sumBot = np.zeros((RD_dB.shape))   # the bottom noise cells
        for idx in range(RD_dB.shape[0]-guardLen-noiseLen):       # for every range bin
            sumBot[idx, :] = np.sum(RD_dB[idx+guardLen:idx+guardLen+noiseLen, :], axis=0)
        for idx in range(guardLen+noiseLen,RD_dB.shape[0]):
            sumTop[idx, :] = np.sum(RD_dB[idx-guardLen-noiseLen:idx-guardLen, :], axis=0)
        
        # one side CFAR for first and last guardLen+noiseLen bins
        sumBot[RD_dB.shape[0]-guardLen-noiseLen:RD_dB.shape[0], :] = sumTop[RD_dB.shape[0]-guardLen-noiseLen:RD_dB.shape[0], :]
        sumTop[0:guardLen+noiseLen, :] = sumBot[0:guardLen+noiseLen, :]
            
        sumTop = sumTop.reshape(sumTop.shape[0], sumTop.shape[1], -1)
        sumBot = sumBot.reshape(sumBot.shape[0], sumBot.shape[1], -1)
        sumTB = np.concatenate((sumTop,sumBot), axis=2)
        sumMin = np.squeeze(np.min(sumTB, axis=2))
        noise = sumMin/noiseLen

        det_points = (RD_dB > noise + self.SUBFRAME_SRR_MIN_SNR_dB).astype(int)
        range_SNR_dB = RD_dB - noise
        return det_points, range_SNR_dB

    
    # -----------------------------------------------------------------------------------------------------------------
    # ljm: Performs on the doppler-CFAR results. Keeps the point with the larest power (dB) in 1x3 sliding windows. 
    # The doppler-CFAR results are wrap padded. If there are 2 or 3 points are detected at a range bin, only 1 is kept.
    #
    # TI: This function pruneToPeaks selects the peaks from within the list of objects detected by CFAR.
    # 
    #   @param[in,out] cfarDetObjIndexBuf  The indices of the detected objects.
    #   @param[in,out] cfarDetObjSNR   The SNR of the detected objects.
    #   @param[in]  numDetObjPerCfar   The number of detected objects.
    #   @param[in]  sumAbs             The sumAbs array on which the CFAR was run.
    # 
    def pruneToPeaks(self, cfarDetPoints, RD_dB):
        # points_pad = np.pad(cfarDetPoints,((0,0),(1,1)), 'wrap')
        # RD = np.pad(RD_dB,((0,0),(1,1)), 'wrap')

        # points_pad2 = np.pad(points_pad, ((0,0), (1,1)), 'constant')
        # points_diff = np.diff(points_pad2, axis=1)
        # start = np.argwhere(points_diff == 1)
        # end = np.argwhere(points_diff == -1)
        # assert len(start) == len(end), "Error in pruneToPeaks!"

        # for i in range(len(start)):
        #     target_segment = RD[start[i,0], start[i,1]:end[i,1]]
        #     maxPos = np.argmax(target_segment)
        #     points_pad[start[i,0], start[i,1]:end[i,1]] = 0
        #     points_pad[start[i,0], start[i,1]+maxPos]   = 1
        # prunePoints = points_pad[:,1:-1]
        prunePoints = np.zeros(cfarDetPoints.shape)
        m, n = cfarDetPoints.shape

        for r in range(m):  # for each range bin
            ptIdx = np.argwhere(cfarDetPoints[r,:] == 1)
            if len(ptIdx) < 2:  # remove single point
                prunePoints[r,ptIdx] = 1
                continue

            pt_dB = RD_dB[r, ptIdx]
            for idx in range(len(pt_dB)): # for each detected point
                if idx == 0:
                    preIdx = len(pt_dB)-1
                else:
                    preIdx = idx-1
                
                if idx == len(pt_dB)-1:
                    nextIdx = 0
                else:
                    nextIdx = idx+1

                if (pt_dB[idx]>pt_dB[preIdx]) and (pt_dB[idx]>pt_dB[nextIdx]):
                    prunePoints[r,ptIdx[idx]] = 1

        return prunePoints

    # -----------------------------------------------------------------------------------------------------------------
    # TI: A slightly weaker implementation of the 'pruneToPeaks'.
    # This variation passes peaks as well as their largest neighbour.
    # 
    def pruneToPeaksOrNeighbourOfPeaks(self, cfarDetPoints, RD_dB):
        prunePoints = np.zeros(cfarDetPoints.shape)
        m, n = cfarDetPoints.shape

        for r in range(m):  # for each range bin
            ptIdx = np.argwhere(cfarDetPoints[r,:] == 1)
            if len(ptIdx) < 3:  # keep single point and only-two points
                prunePoints[r,ptIdx] = 1
                continue

            pt_dB = RD_dB[r, ptIdx]
            for idx in range(len(pt_dB)): # for each detected point
                if idx == 0:
                    preIdx = len(pt_dB)-1
                else:
                    preIdx = idx-1
                
                if preIdx == 0:
                    prepreIdx = len(pt_dB)-1
                else:
                    prepreIdx = preIdx-1
                
                if idx == len(pt_dB)-1:
                    nextIdx = 0
                else:
                    nextIdx = idx+1

                if nextIdx == len(pt_dB)-1:
                    nextnextIdx = 0
                else:
                    nextnextIdx = nextIdx+1
                
                is_peak = (pt_dB[idx]>pt_dB[preIdx]) and (pt_dB[idx]>pt_dB[nextIdx])
                is_neighbourOfPeakNext = (pt_dB[nextnextIdx] < pt_dB[idx]) and (pt_dB[preIdx] < pt_dB[idx])
                is_neighbourOfPeakPrev = (pt_dB[nextIdx] < pt_dB[idx]) and (pt_dB[prepreIdx] < pt_dB[idx])

                if is_peak or is_neighbourOfPeakNext or is_neighbourOfPeakPrev:
                    prunePoints[r,ptIdx[idx]] = 1

        return prunePoints

    # -----------------------------------------------------------------------------------------------------------------
    # ljm: This function does not group detected points, but check to make sure a detected point has the largest 
    # power in it's 1x3 neighborhood

    # TI: Peak grouping
    #     Another pruning step for the point cloud because we haven't made
    #     sure that the objects are peaks in doppler.

    # TI: The function groups neighboring peaks (only in the doppler direction) into one. For each
    # detected peak the function checks if the peak is greater than its neighbors. If this is true,
    # the peak is copied to the output list of detected objects. The neighboring peaks that are used 
    # for checking are taken from the detection matrix and copied into 1x3 kernel regardless of
    # whether they are CFAR detected or not.
    def cfarPeakGroupingAlongDoppler(self, cfarDetPoints, RD_dB):
        prunePoints = np.zeros(cfarDetPoints.shape)
        m, n = cfarDetPoints.shape

        for r in range(m):  # for each range bin
            ptIdx = np.argwhere(cfarDetPoints[r,:] == 1)
            for idx in ptIdx: # for each detected point
                if idx == 0:
                    preIdx = n-1
                else:
                    preIdx = idx-1
                
                if idx == n-1:
                    nextIdx = 0
                else:
                    nextIdx = idx+1

                if (RD_dB[r,idx]>RD_dB[r,preIdx]) and (RD_dB[r,idx]>RD_dB[r,nextIdx]):
                    prunePoints[r,idx] = 1

        return prunePoints

    # -----------------------------------------------------------------------------------------------------------------
    # Merge the doppler and range CFAR results
    def merge_Dopper_Range_CFAR(self, doppler_CFAR_det_points, range_CFAR_det_points):
        results = np.zeros(doppler_CFAR_det_points.shape)
        
        pos = np.argwhere(doppler_CFAR_det_points==1)
        doppler_bins = np.unique(pos[:,1])

        results[:,doppler_bins] = range_CFAR_det_points[:,doppler_bins]

        return results



# -----------------------------------------------------------------------------------------------------------------    
class RadarSignalProcessing():
    def __init__(self,radar_params):

        self.radar_params = radar_params
            
        # Radar parameters
        self.numSamplePerChirp  = radar_params["range_size"]
        # self.numRxPerChip       = 4
        self.numChirps          = radar_params["doppler_size"]
        self.numRxAnt           = radar_params["num_Rx"]
        self.numTxAnt           = radar_params["num_Tx"]
        # self.numReducedDoppler  = 16
        # self.numChirpsPerLoop   = 16
        self.numSampleAzimuth   = radar_params["azimuth_size"]

        self.CFAR_fct = CA_CFAR(win_param=(9,9,3,3), threshold=2, rd_size=(self.numSamplePerChirp,16))

        self.multiObjBeamFormingEnabled = radar_params["multiObjBeamFormingEnabled"]    # enable to detect the second peak in azimuth
        self.multiPeakThrsScal          = radar_params["multiPeakThrsScal"]             # the portion of the amplitidue (dB) of the second peak to largest peak
            
        # Build hamming window table to reduce side lobs
        hanningWindowRange      = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numSamplePerChirp ))/(self.numSamplePerChirp -1))))
        hanningWindowDoppler    = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numChirps ))/(self.numChirps -1))))
        hanningWindowAzimuth    = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numSampleAzimuth ))/(self.numSampleAzimuth -1))))
        # hanningWindowRange      = (0.5 - 0.5*np.cos(((2*math.pi*np.arange(self.numSamplePerChirp ))/(self.numSamplePerChirp -1))))
        # hanningWindowDoppler    = (0.5 - 0.5*np.cos(((2*math.pi*np.arange(self.numChirps ))/(self.numChirps -1))))
        # hanningWindowAzimuth    = (0.5 - 0.5*np.cos(((2*math.pi*np.arange(self.numSampleAzimuth ))/(self.numSampleAzimuth -1))))
        self.range_fft_coef     = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange,1), repeats=self.numChirps, axis=1),2)
        self.doppler_fft_coef   = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=self.numSamplePerChirp, axis=0),2)
        self.azimuth_fft_coef   = np.expand_dims(np.repeat(np.expand_dims(hanningWindowAzimuth, 1).transpose(), repeats=self.numSamplePerChirp, axis=0),2)
        self.azimuth_fft_coef   = self.azimuth_fft_coef.transpose((0,2,1))

        # TI: MIMO radar application report
        # Prior to applying angle FFT, a Doppler correction step must be performed in order to correct for any velocity induced phase change.
        # Reference, Xinrong Li, et al., Signal processing for TDM MIMO FMCW MM radar sensors, IEEE Access, 9:167959, 2021
        # Eq. (55)
        # RD = RD*exp(-2*PI*j*(nc*nt)/(Nc*Nt))
        # Nc: number of chirps
        # nc: the nc-th chirp, 0<=nr<Nc
        # nt: the nt-th transmitter, 0<=nt<Nt
        Nt = self.numTxAnt
        Nr = self.numRxAnt
        Nc = self.numChirps

        nc = np.arange(Nc).reshape((1, Nc)) - Nc/2  # 1x64
        nc = np.expand_dims(np.repeat(nc, self.numSamplePerChirp, axis=0), 2)   # 256x64x1
        nc = np.repeat(nc, Nt*Nr, axis=2)           # 256x64x8

        nt = np.arange(Nt*Nr).reshape((1, -1))      # 1x8: 0,1,..., 7
        nt = np.floor(nt/4)                         # 1x8: 0,0,0,0, 1,1,1,1
        nt = np.expand_dims(np.repeat(nt, self.numSamplePerChirp, axis=0), 2)   # 256x8x1
        nt = nt.transpose((0,2,1))                  # 256x1x8

        idx = np.multiply(nc,nt)
        self.phase_correction_coef = np.exp(-2j*np.pi*idx/(Nc*Nt))

    def get_RD(self, ADC, staticDepression=False):
        # 1. Decode the input ADC stream to buld radar complex frames
        # complex_adc = self.__build_radar_frame(adc0,adc1,adc2,adc3)
        complex_adc = ADC
    
        # 2- Remoce DC offset
        complex_adc = complex_adc - np.mean(complex_adc, axis=0)

        # 3- Range FFT
        range_fft = mkl_fft.fft(np.multiply(complex_adc, self.range_fft_coef), self.numSamplePerChirp, axis=0)
        
        # https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html
        # scale the output for power specturm estimation
        # range_fft = range_fft/(np.sqrt(2*np.pi*self.numSamplePerChirp))

        # 4- Doppler FFT
        # RD_spectrums = mkl_fft.fft(np.multiply(range_fft, self.doppler_fft_coef), self.numChirps,axis=1)  
        if staticDepression:
            range_fft = range_fft - np.expand_dims(np.mean(range_fft, axis=1), 2).transpose((0,2,1))          # static clutter depression
        RD_spectrums = mkl_fft.fft(range_fft, self.numChirps,axis=1)    # do not need windowing for Doppler FFT
        RD_spectrums = np.fft.fftshift(RD_spectrums, axes=1)

        # scale the output for power specturm estimation, seems that RADDet does not do it. Jimin
        # RD_spectrums = RD_spectrums/(np.sqrt(2*np.pi*self.numChirps))

        # flip the range, following RADDet's data format
        RD_spectrums = np.flip(RD_spectrums, axis=0)

        # Doppler phase correction, seems that RADDet does not do it. Jimin
        RD_spectrums = np.multiply(RD_spectrums, self.phase_correction_coef)

        return RD_spectrums, range_fft

    def get_RAD(self,RD_spectrums):

        # zero padding
        RD = np.zeros((RD_spectrums.shape[0], RD_spectrums.shape[1], self.numSampleAzimuth), dtype=RD_spectrums.dtype)
        RD[:,:,0:RD_spectrums.shape[2]] = RD_spectrums

        # RAD_spectrums = mkl_fft.fft(np.multiply(RD,self.azimuth_fft_coef), self.numSampleAzimuth, axis=2) 
        RAD_spectrums = mkl_fft.fft(RD, self.numSampleAzimuth, axis=2)
        RAD_spectrums = np.fft.fftshift(RAD_spectrums, axes=2)
        
        # scale the output for power specturm estimation
        # RAD_spectrums = RAD_spectrums/(np.sqrt(2*np.pi*self.numSampleAzimuth))

        # reshape RAD: range-azimuth-chirp=256x256x64
        RAD_spectrums = RAD_spectrums.transpose((0,2,1))  

        # RADDet did this ????????
        RAD_spectrums = np.flip(RAD_spectrums, axis=1)  

        return RAD_spectrums

    # get the point clound from the CFAR result and the RAD
    def get_PCL_RADDet(self, cfarDetPoints, doppler_SNR_dB, range_SNR_dB, RAD):
        
        # Tow to calcualte the azimuth: file:///D:/ti/mmwave_sdk_02_01_00_04/packages/ti/demo/xwr16xx/mmw/docs/doxygen/html/index.html
        # The phase difference between Rx antenna: w = 2pi/N * Kmax, Kmax is the index of the peak of the azimuth FFT in the range of [-N/2, N/2 - 1]
        # w = pi*sin(theta)
        # so thet = arcsin(2/N*Kmax)

        range_resolution    = self.radar_params['range_resolution']
        angular_resolution  = self.radar_params['angular_resolution']
        velocity_resolution = self.radar_params['velocity_resolution']
        Nr                  = self.radar_params['range_size']
        Nc                  = self.radar_params['doppler_size']
        Na                  = self.radar_params['azimuth_size']
        
        points = np.argwhere(cfarDetPoints == 1)

        objProperties = ('rangeIdx', 'dopplerIdx', 'azimuthIdx', 'range', 'speed', 'azimuth', 'peakVal','rangeSNRdb', 'dopplerSNRdb' )
        df_objects = pd.DataFrame(columns=objProperties)
        for i in range(len(points)):
            azimuthMagdB = 20*np.log10(np.abs(RAD[points[i,0], :, points[i,1]])).reshape(1,-1)
            
            maxPeak = np.max(azimuthMagdB)
            maxPeakIdx = np.argmax(azimuthMagdB)

            # range direction is flipped
            # azimuth = np.arcsin(2*(maxPeakIdx-Na/2)/Na)*180/np.pi
            obj = { 'rangeIdx':     points[i,0], \
                    'dopplerIdx':   points[i,1], \
                    'azimuthIdx':   maxPeakIdx, \
                    'range':        (Nr-1-points[i,0])*range_resolution, \
                    'speed':        (points[i,1]-Nc/2) * velocity_resolution, \
                    'azimuth':      np.arcsin( (maxPeakIdx * (2*np.pi/Na) - np.pi) / (2*np.pi*0.5*self.radar_params["config_frequency"]/self.radar_params["designed_frequency"])) *180/np.pi, \
                    'peakVal':      maxPeak, \
                    'rangeSNRdb':   range_SNR_dB[points[i,0], points[i,1]], \
                    'dopplerSNRdb': doppler_SNR_dB[points[i,0], points[i,1]]
            }
            obj = pd.DataFrame(obj, index=[0])
            df_objects = pd.concat([df_objects, obj], ignore_index=True)

        return df_objects


    # function for CARRADA dataset, does not work for RADDet dataset
    def get_PCL_CARRADA(self,RD_spectrums):
        # 1- Compute power spectrum
        power_spectrum = np.sum(np.abs(RD_spectrums),axis=2)

        # 2- Apply CFAR
        # But because Tx are phase shifted of DopplerShift=16, then reduce spectrum to MaxDoppler/16 on Doppler axis
        reduced_power_spectrum = np.sum(power_spectrum.reshape(512,16,16),axis=1)
        peaks = self.CFAR_fct(reduced_power_spectrum)
        RangeBin,DopplerBin_conv = np.where(peaks>0)

        # 3- Need to find TX0 position to rebuild the MIMO spectrum in the correct order
        DopplerBin_candidates = self.__find_TX0_position(power_spectrum, RangeBin, DopplerBin_conv)
        RangeBin_candidates = [[i] for i in RangeBin]
        doppler_indexes = []
        for doppler_bin in DopplerBin_candidates:
            DopplerBinSeq = np.remainder(doppler_bin+ self.dividend_constant_arr, self.numChirps)
            DopplerBinSeq = np.concatenate([[DopplerBinSeq[0]],DopplerBinSeq[5:]]).astype('int')
            doppler_indexes.append(DopplerBinSeq)
            

        # 4- Extract and reshape the Rx * Tx matrix into the MIMO spectrum
        MIMO_Spectrum = RD_spectrums[RangeBin_candidates,doppler_indexes,:].reshape(len(DopplerBin_candidates),-1)
        MIMO_Spectrum = np.multiply(MIMO_Spectrum,self.window)
        
        # 5- AoA: maker a cross correlation between the recieved signal vs. the calibration matrix 
        # to identify azimuth and elevation angles
        ASpec=np.abs(self.CalibMat@MIMO_Spectrum.transpose())
        
        # 6- Extract maximum per (Range,Doppler) bins
        x,y = np.where(np.isnan(ASpec))
        ASpec[x,y] = 0
        az,el = np.unravel_index(np.argmax(ASpec,axis=0),(self.AoA_mat['Signal'].shape[0],self.AoA_mat['Signal'].shape[2]))
        az = np.deg2rad(self.AoA_mat['Azimuth_table'][az])
        el = np.deg2rad(self.AoA_mat['Elevation_table'][el])
        
        RangeBin = RangeBin/self.numSamplePerChirp*103.
        
        return np.vstack([RangeBin,DopplerBin_candidates,az,el]).transpose()
        
