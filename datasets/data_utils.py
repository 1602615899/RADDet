import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

def readRAD(filename):
    """ read input RAD matrices """
    if os.path.exists(filename):
        return np.load(filename)
    else:
        return None
################ coordinates transformation ################
def cartesianToPolar(x, y):
    """ Cartesian to Polar """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polarToCartesian(rho, phi):
    """ Polar to Cartesian """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

################ functions of RAD processing ################

def normalize_data(data):
    '''
    normalize data to [0, 1]
    '''
    data_min = data.min()
    data_max = data.max()
    
    normalized_data = (data - data_min) / (data_max - data_min) 
    
    return normalized_data

def complexTo2Channels(target_array):
    """ transfer complex a + bi to [a, b]"""
    assert target_array.dtype == np.complex64
    ### NOTE: transfer complex to (magnitude) ###
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array

def preprocess_data(target_array):
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array

def getMagnitude(target_array, power_order=2):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array

def getSumDim(target_array, target_axis):
    """ sum up one dimension """
    output = np.sum(target_array, axis=target_axis)
    return output 

def switchCols(target_array, cols):
    """ switch columns """
    assert isinstance(cols, tuple) or isinstance(cols, list)
    assert len(cols) == 2
    assert np.max(cols) <= target_array.shape[-1] - 1
    cols = np.sort(cols)
    output_axes = []
    for i in range(target_array.shape[-1]):
        if i == cols[0]:
            idx = cols[1]
        elif i == cols[1]:
            idx = cols[0]
        else:
            idx = i
        output_axes.append(idx)
    return target_array[..., output_axes]

def switchAxes(target_array, axes):
    """ switch axes """
    assert isinstance(axes, tuple) or isinstance(axes, list)
    assert len(axes) == 2
    assert np.max(axes) <= len(target_array.shape) - 1
    return np.swapaxes(target_array, axes[0], axes[1])

def norm2Image(array):
    """ normalize to image format (uint8) """
    norm_sig = plt.Normalize()
    img = plt.cm.viridis(norm_sig(array))
    img *= 255.
    img = img.astype(np.uint8)
    return img


################ functions of RAD visualization ################

def imgPlot(img, ax, cmap=None, alpha=1, title=None):
    """ image plotting (customized when plotting RAD) """
    ax.imshow(img, cmap=cmap, alpha=alpha)
    if title == "RD":
        title = "Range-Doppler"
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_xticklabels([-13, -6.5, 0, 6.5, 13])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("velocity (m/s)")
        ax.set_ylabel("range (m)")
    elif title == "RA":
        title = "Range-Azimuth"
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels([-85.87, -42.93, 0, 42.93, 85.87])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("angle (degrees)")
        ax.set_ylabel("range (m)")
    elif title == "DA":
        title = "Doppler-Azimuth"
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels([-85.87, -42.93, 0, 42.93, 85.87])
        ax.set_yticks([0, 16, 32, 48, 63])
        ax.set_yticklabels([-13, -6.5, 0, 6.5, 13])
        ax.set_xlabel("angle (degrees)")
        ax.set_ylabel("velocity (m/s)")
    elif title == "Cartesian":
        ax.set_xticks([0, 128, 256, 384, 512])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    else:
        ax.axis('off')
    if title is not None:
        ax.set_title(title)

def plot_RAD(RAD_data, raw=False, colorbar_flag=True, suptitle=None, save_path=None,  random_idx_ra = None, random_idx = None):
    """ plot RAD data , [R, A, D]"""
    if raw:
        RA = getLog(getSumDim(getMagnitude(RAD_data, power_order=2), target_axis=-1), scalar=10., log_10=True)
        RD = getLog(getSumDim(getMagnitude(RAD_data, power_order=2), target_axis=1), scalar=10., log_10=True)
        DA = getLog(getSumDim(getMagnitude(RAD_data, power_order=2).transpose(2, 0, 1), target_axis=1), scalar=10., log_10=True)
    else:
        # mean
        RA = RAD_data.take(indices=random_idx_ra, axis=-1)
        RD = RAD_data.take(indices=random_idx, axis=1)
        DA = RAD_data.transpose(2, 0, 1).take(indices=random_idx, axis=1)

        # RA = getSumDim(RAD_data, target_axis=-1)
        # RD = getSumDim(RAD_data, target_axis=1)
        # DA = getSumDim(RAD_data.transpose(2, 0, 1), target_axis=1)
        # max
        # RA = RAD_data.max(axis = -1)
        # RD = RAD_data.max(axis = 1)
        # DA = RAD_data.transpose(2, 0 ,1).max(axis = 1)
    
    RA_img = norm2Image(RA)[..., :3]
    RD_img = norm2Image(RD)[..., :3]
    DA_img = norm2Image(DA)[..., :3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(suptitle, fontsize=16)
    # axes[0].imshow(RA_img)
    # axes[1].imshow(RD_img)
    # axes[2].imshow(DA_img)
    imgPlot(RA_img, axes[0], title="RA")        # cmap: viridis hot
    imgPlot(RD_img, axes[1], title="RD")
    imgPlot(DA_img, axes[2], title="DA")

    plt.tight_layout()
    # plt.savefig("./test.png")

    if colorbar_flag:
        cmap = plt.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)  # 设置 colorbar 范围
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), 
                            ax=axes[-1], orientation='vertical', 
                            fraction=0.05, pad=0.05,shrink=0.8,
                            # location = 'left'
                            ) # pad 控制 colorbar 与图片的距离，shrink 控制 colorbar 与图片的比例，fraction 控制 colorbar 在整个图形中所占的相对大小
        # colorbar 都一样的 从蓝到黄，-1到1
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()  # 保存后关闭图形
    else:
        plt.show()


