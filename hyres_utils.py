import numpy as np
import h5py 
import os
import glob
import cv2
import tifffile as tiff
import yaml
from natsort import natsorted, ns
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from tqdm.auto import tqdm


def _dump_array_yaml(array, path):
    """Write an array as a plain YAML list (readable, NumPy-version independent)."""
    with open(path, 'w') as f:
        yaml.safe_dump(np.asarray(array, dtype=np.float64).tolist(), f)


def _load_array_yaml(path):
    """Read minmax/features YAML: plain lists, or legacy files holding pickled NumPy arrays."""
    with open(path) as f:
        return np.asarray(yaml.load(f, Loader=yaml.Loader), dtype=np.float64)

def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn) / mx) * 255.0
    return np.round(I).astype(np.uint8)

def downsample(filelist, directory, savedir):
    # bicubic downsample the images 4 times on both axes
    filelist = os.listdir(filelist)
    directory = directory
    savedir = savedir
    for i in tqdm(filelist):
        img = cv2.imread(directory + i)
        img_resized = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(savedir + i, img_resized)

def img2gray(arr):
    # intensity to 0-255
    Imax = np.max(arr)
    Imin = np.min(arr)
    k = 255.0 / (Imax - Imin)
    arr = k * (arr - Imin)
    return np.round(arr, 2)


def preselect(path):
    f1 = h5py.File(path, 'r')

    # 3D MS image data
    liver1 = f1.get('counts_mat')
    liver1 = np.array(liver1)
    z = liver1.shape[1]
    dimxy = f1.get('dim_xy')
    dimxy = np.array(dimxy)
    liver1 = np.reshape(liver1, (dimxy[1], dimxy[0], z))

    # Get the background noise
    hsnr = []
    lsnr = []
    snrlist = []
    mean_norm = []
    bg_list = []
    for i in range(z):
        gray = img2gray(liver1[:, :, i])
        mean = np.mean(gray)
        bg_list.append(mean)
    mi = np.argmin(bg_list)  # bg = the lowest mean intensity under gray image
    bg_gray = img2gray(liver1[:, :, mi])
    bg = np.mean(bg_gray)

    # Calculate SNR and filter out super low snr
    for i in range(z):
        gray = img2gray(liver1[:, :, i])
        mean = np.mean(gray)
        snr = mean / bg
        snr = 20 * np.log10(snr)
        snrlist.append(snr)
        mean_norm.append(mean)

    snrlist = np.array(snrlist)
    snr_mean = np.mean(snrlist)
    snr_med = np.median(snrlist)
    snr_standard = snr_mean

    for i in range(z):
        if snrlist[i] > snr_standard * 1:
            hsnr.append(i)
        else:
            lsnr.append(i)
    # Save the normalized gray image data
    for i in range(liver1.shape[2]):
        liver1[:, :, i] = img2gray(liver1[:, :, i])

    hr = np.delete(liver1, lsnr, axis=2)

    return hr

def noise_adding(path, savedir):
    # add artificial noise to image
    p = 0.5
    level = 50
    img_arr = np.load(path)

    lr0 = np.random.poisson(img_arr)
    gr0 = level * np.random.normal(0, 1, img_arr.shape) + img_arr
    nr0 = lr0 * p + gr0 * (1 - p)
    nr0 = np.clip(nr0, 0.0, 255.0)
    np.save(savedir, nr0)



def _imzml_to_datacube(path, features=None, tol=10, tol_unit='ppm',
                       bin_width=None, dtype=np.float32):
    """
    Read an imzML file into a (n_features x n_pixels) datacube.

    - Continuous imzML: uses the shared m/z axis directly (features=None).
    - Processed imzML: needs either
        * features (list/array of target m/z) with tol in 'ppm' or 'da', or
        * bin_width (Da) for regular binning between global min/max m/z.
    Returns datacube, features, dims=[x, y] (same layout as the h5 branch).
    """
    p = ImzMLParser(path)
    coords = np.asarray(p.coordinates)[:, :2].astype(int)       # 1-based (x, y)
    x0, y0 = coords[:, 0].min(), coords[:, 1].min()
    dims = [int(coords[:, 0].max() - x0 + 1), int(coords[:, 1].max() - y0 + 1)]
    flat_idx = (coords[:, 1] - y0) * dims[0] + (coords[:, 0] - x0)  # row-major
    n_pix = dims[0] * dims[1]
    continuous = len(set(p.mzOffsets)) == 1

    # ---- continuous mode, native m/z axis ----
    if features is None and bin_width is None:
        if not continuous:
            raise ValueError('Processed-mode imzML: supply `features` (+ tol) '
                             'or `bin_width` to define a common m/z axis.')
        features = np.asarray(p.getspectrum(0)[0], dtype=np.float64)
        datacube = np.zeros((features.size, n_pix), dtype=dtype)
        for i, pix in enumerate(flat_idx):
            datacube[:, pix] = p.getspectrum(i)[1]
        return datacube, features, dims

    # ---- regular binning ----
    if features is None:
        lo, hi = np.inf, -np.inf
        for i in range(len(p.coordinates)):
            mzs = p.getspectrum(i)[0]
            if mzs.size:
                lo, hi = min(lo, mzs.min()), max(hi, mzs.max())
        edges = np.arange(lo, hi + bin_width, bin_width)
        features = edges[:-1] + bin_width / 2
        nf = features.size
        datacube = np.zeros((nf, n_pix), dtype=dtype)
        for i, pix in enumerate(flat_idx):
            mzs, ints = p.getspectrum(i)
            b = np.clip(((mzs - lo) // bin_width).astype(int), 0, nf - 1)
            datacube[:, pix] = np.bincount(b, weights=ints, minlength=nf)
        return datacube, features, dims

    # ---- user-defined feature list with tolerance ----
    features = np.sort(np.asarray(features, dtype=np.float64))
    nf = features.size
    datacube = np.zeros((nf, n_pix), dtype=dtype)
    for i, pix in enumerate(flat_idx):
        mzs, ints = p.getspectrum(i)
        if mzs.size == 0:
            continue
        j = np.clip(np.searchsorted(features, mzs), 1, nf - 1)
        left, right = features[j - 1], features[j]
        nearest = np.where(np.abs(mzs - left) <= np.abs(mzs - right), j - 1, j)
        if nf == 1:
            nearest[:] = 0
        dev = np.abs(mzs - features[nearest])
        lim = features[nearest] * tol * 1e-6 if tol_unit.lower() == 'ppm' else tol
        m = dev <= lim
        datacube[:, pix] = np.bincount(nearest[m], weights=ints[m], minlength=nf)
    return datacube, features, dims


def _datacube_to_imzml(path, datacube, features, dims, skip_empty=False,
                       polarity=None):
    """Write a (n_features x n_pixels) datacube as continuous imzML (+ .ibd)."""
    features = np.asarray(features, dtype=np.float64)
    order = np.argsort(features)                      # imzML expects ascending m/z
    features, datacube = features[order], datacube[order, :]
    kw = {'polarity': polarity} if polarity else {}
    with ImzMLWriter(path, mode='continuous', mz_dtype=np.float64,
                     intensity_dtype=np.float32, **kw) as w:
        for pix in range(dims[0] * dims[1]):
            spec = datacube[:, pix]
            if skip_empty and not np.any(spec):
                continue
            x, y = pix % dims[0] + 1, pix // dims[0] + 1  # 1-based coordinates
            w.addSpectrum(features, spec.astype(np.float32), (x, y))


def hyres_generator(filename, output_dir, file_format='h5',
                    features=None, tol=10, tol_unit='ppm', bin_width=None):
    """
    file_format: 'h5' or 'imzml'.
    For processed-mode imzML pass `features` (+ tol/tol_unit) or `bin_width`.
    """
    print(filename)
    in_path = os.path.join(output_dir, filename)
    fmt = file_format.lower()

    if fmt == 'h5':
        with h5py.File(in_path, 'r') as h5:
            datacube = h5['datacube'][()]
            dims = np.array(h5['dims']).astype(int)
            features = np.array(h5['mz'])
    elif fmt == 'imzml':
        datacube, features, dims = _imzml_to_datacube(
            in_path, features=features, tol=tol, tol_unit=tol_unit,
            bin_width=bin_width)
    else:
        raise ValueError(f"Unsupported file_format '{file_format}'")

    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0])
    os.makedirs(output_dir, exist_ok=True)

    minmax = np.empty([2, len(features)])
    for n in range(len(features)):
        ion_image = np.asarray(datacube[n, :], dtype=np.float64)
        ion_image = ion_image.reshape((int(dims[1]), int(dims[0])))
        minmax[0, n] = ion_image.min()
        minmax[1, n] = ion_image.max()
        rng = minmax[1, n] - minmax[0, n]
        ion_image = (ion_image - minmax[0, n]) / rng if rng > 0 else np.zeros_like(ion_image)
        rgb = (np.dstack((ion_image, ion_image, ion_image)) * 65535.999).astype(np.uint16)
        tiff.imwrite(os.path.join(output_dir, filename + ' feature %d' % n + '.tif'), rgb)

    _dump_array_yaml(minmax, os.path.join(output_dir, 'minmax.yaml'))
    _dump_array_yaml(features, os.path.join(output_dir, 'features.yaml'))
    return minmax, features


def hyres_degenerator(input_dir, file_format='h5', save=True,
                      skip_empty=False, polarity=None):
    """
    Rebuild the datacube from the TIFF stack and save as 'h5' or 'imzml'.
    """
    fmt = file_format.lower()
    minmax = _load_array_yaml(os.path.join(input_dir, 'minmax.yaml'))
    features = _load_array_yaml(os.path.join(input_dir, 'features.yaml'))

    tifs = natsorted(glob.glob(os.path.join(input_dir, '*.tif')), alg=ns.IGNORECASE)
    if len(tifs) != len(features):
        raise ValueError(f'{len(tifs)} TIFFs but {len(features)} features in {input_dir}')
    first = tiff.imread(tifs[0])
    dims = [first.shape[1], first.shape[0]]                      # [x, y]
    datacube = np.empty([len(features), dims[0] * dims[1]])
    print(datacube.shape)

    for n, file in enumerate(tifs):
        image_n = np.asarray(tiff.imread(file), dtype=np.float64)
        if image_n.ndim == 3:                                    # RGB stack -> single channel
            image_n = image_n.mean(axis=2)
        image_n = image_n / 65535.0                              # back to [0, 1]
        image_n = image_n * (minmax[1, n] - minmax[0, n]) + minmax[0, n]
        datacube[n, :] = image_n.reshape(dims[0] * dims[1])

    if save:
        if fmt == 'h5':
            with h5py.File(os.path.join(input_dir, 'datacube_HyReS.h5'), 'w') as f:
                f.create_dataset('datacube', data=datacube)
                f.create_dataset('dimension', data=dims)
                f.create_dataset('features', data=features)
        elif fmt == 'imzml':
            _datacube_to_imzml(os.path.join(input_dir, 'datacube_HyReS.imzML'),
                               datacube, features, dims,
                               skip_empty=skip_empty, polarity=polarity)
        else:
            raise ValueError(f"Unsupported file_format '{file_format}'")
    return datacube, features, dims
