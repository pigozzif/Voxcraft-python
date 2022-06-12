import numpy as np
import itertools
import re
import zlib


def identity(x):
    return x


def sigmoid(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0


def positive_sigmoid(x):
    return (1 + sigmoid(x)) * 0.5


def rescaled_positive_sigmoid(x, x_min=0, x_max=1):
    return (x_max - x_min) * positive_sigmoid(x) + x_min


def inverted_sigmoid(x):
    return sigmoid(x) ** -1


def neg_abs(x):
    return -np.abs(x)


def neg_square(x):
    return -np.square(x)


def sqrt_abs(x):
    return np.sqrt(np.abs(x))


def neg_sqrt_abs(x):
    return -sqrt_abs(x)


def neg_sign(x):
    return -np.sign(x)


def mean_abs(x):
    return np.mean(np.abs(x))


def std_abs(x):
    return np.std(np.abs(x))


def count_positive(x):
    return np.sum(np.greater(x, 0))


def count_negative(x):
    return np.sum(np.less(x, 0))


def proportion_equal_to(x, keys):
    return np.mean(count_occurrences(x, keys))


def normalize(x):
    x -= np.min(x)
    x /= np.max(x)
    x = np.nan_to_num(x)
    x *= 2
    x -= 1
    return x


def compressed_size(a):
    return len(zlib.compress(a))


def bootstrap_ci(a, func, n=5000, ci=95):
    stats = func(np.random.choice(a, (n, len(a))), axis=1)
    lower = np.percentile(stats, 100 - ci)
    upper = np.percentile(stats, ci)
    return lower, upper


def vox_id_from_xyz(x, y, z, size):
    return z * size[0] * size[1] + y * size[0] + x


def vox_xyz_from_id(idx, size):
    z = idx / (size[0] * size[1])
    y = (idx - z * size[0] * size[1]) / size[0]
    x = idx - z * size[0] * size[1] - y * size[0]
    return x, y, z


def convert_voxelyze_index(v_index, spacing=0.010, start_pos=0.005):
    return int(v_index / spacing - start_pos)


def resize_voxarray(a, pad=2, const=0):
    if isinstance(pad, int):
        n_pad = ((pad, pad),) * 3  # (n_before, n_after) for each dimension
    else:
        n_pad = pad
    return np.pad(a, pad_width=n_pad, mode='constant', constant_values=const)


def get_outer_shell(a):
    x, y, z = a.shape
    return [a[0, :, :], a[x - 1, :, :], a[:, 0, :], a[:, y - 1, :], a[:, :, 0], a[:, :, z - 1]]


def get_outer_shell_complements(a):
    x, y, z = a.shape
    return [a[1:, :, :], a[:x - 1, :, :], a[:, 1:, :], a[:, :y - 1, :], a[:, :, 1:], a[:, :, :z - 1]]


def trim_voxarray(a):
    new = np.array(a)
    done = False
    while not done:
        outer_slices = get_outer_shell(new)
        inner_slices = get_outer_shell_complements(new)
        for i, o in zip(inner_slices, outer_slices):
            if np.sum(o) == 0:
                new = i
                break

        voxels_in_shell = [np.sum(s) for s in outer_slices]
        if 0 not in voxels_in_shell:
            done = True

    return new


def get_depths_of_material_from_shell(a, mat):
    tmp = a
    depth = [0] * 6
    done = [False] * 6

    while not np.all(done):
        shell = get_outer_shell(tmp)
        mat_in_shell = [mat in s for s in shell]

        for n in range(6):
            if not done[n]:
                if mat_in_shell[n]:
                    done[n] = True
                else:
                    inner_slices = get_outer_shell_complements(tmp)
                    tmp = inner_slices[n]
                    depth[n] += 1

    return depth


def get_mat_span(a, mat):
    depths = get_depths_of_material_from_shell(a, mat)
    return [a.shape[0] - depths[1] - depths[0], a.shape[1] - depths[3] - depths[2], a.shape[2] - depths[5] - depths[4]]


def reorder_vxa_array(a, size):
    anew = np.empty(size)
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                anew[x, y, z] = a[z, y * size[0] + x]
    return anew


def array_to_vxa(a):
    anew = np.empty((a.shape[2], a.shape[1] * a.shape[0]))
    for z in range(a.shape[2]):
        for y in range(a.shape[1]):
            for x in range(a.shape[0]):
                anew[z, y * a.shape[0] + x] = a[x, y, z]
    return anew


def xml_format(tag):
    """Ensures that tag is encapsulated inside angle brackets."""
    if tag[0] != "<":
        tag = "<" + tag
    if tag[-1:] != ">":
        tag += ">"
    return tag


def get_data_from_xml_line(line, tag, dtype=float):
    try:
        return dtype(line[line.find(tag) + len(tag):line.find("</" + tag[1:])])
    except ValueError:
        start = line.find(">")
        end = line.find("</")
        return dtype(line[start + 1:end])


def natural_sort(l, reverse):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key, reverse=reverse)


def find_between(string, start, end):
    start = string.index(start) + len(start)
    end = string.index(end, start)
    return string[start:end]


def replace_text_in_file(filename, replacements_dict):
    lines = []
    with open(filename) as infile:
        for line in infile:
            for original, target in replacements_dict.iteritems():
                line = line.replace(original, target)
            lines.append(line)
    with open(filename, 'w') as outfile:
        for line in lines:
            outfile.write(line)


def dominates(ind1, ind2, attribute_name, maximize):
    """Returns True if ind1 dominates ind2 in a shared attribute."""
    if maximize:
        return getattr(ind1, attribute_name) > getattr(ind2, attribute_name)
    else:
        return getattr(ind1, attribute_name) < getattr(ind2, attribute_name)


def count_occurrences(x, keys):
    """Count the total occurrences of any keys in x."""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    active = np.zeros_like(x, dtype=np.bool)
    for a in keys:
        active = np.logical_or(active, x == a)
    return active.sum()


def count_neighbors(output_state, mask=None):
    """Count neighbors of each 3D element after applying boolean mask.
    Parameters
    ----------
    output_state : numpy.ndarray
        Network output
    mask : bool mask
        Threshold function applied to output_state
    Returns
    -------
    num_of_neighbors : list
        Count of True elements surrounding an individual in 3D space.
    """
    if mask is None:
        def mask(u): return np.greater(u, 0)

    presence = mask(output_state)
    voxels = list(itertools.product(*[range(x) for x in output_state.shape]))
    num_neighbors = [0 for _ in voxels]

    for idx, (x, y, z) in enumerate(voxels):
        for neighbor in [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]:
            if neighbor in voxels:
                num_neighbors[idx] += presence[neighbor]

    return num_neighbors


def get_neighbors(a):
    b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
    neigh = np.concatenate((
        b[2:, 1:-1, 1:-1, None], b[:-2, 1:-1, 1:-1, None],
        b[1:-1, 2:, 1:-1, None], b[1:-1, :-2, 1:-1, None],
        b[1:-1, 1:-1, 2:, None], b[1:-1, 1:-1, :-2, None]), axis=3)
    return neigh


def quadruped(shape, cut_leg=None, half_cut=False, double_cut=False, pad=0, height_above_waist=0,
              x_depth_beyond_midline=-1, y_depth_beyond_midline=-1, mat=1, patch_mat=2):
    bot = np.ones(shape, dtype=int) * mat
    adj0 = shape[0] % 2 == 0
    adj1 = shape[1] % 2 == 0

    # legs to keep
    front_right_leg = [range(shape[2] / 2 + height_above_waist),
                       range(shape[0] / 2 - x_depth_beyond_midline, shape[0]),
                       range(shape[1] / 2 + 1 + y_depth_beyond_midline - adj1)]

    front_left_leg = [range(shape[2] / 2 + height_above_waist),
                      range(shape[0] / 2 + 1 + x_depth_beyond_midline - adj0),
                      range(shape[1] / 2 + 1 + y_depth_beyond_midline - adj1)]

    back_right_leg = [range(shape[2] / 2 + height_above_waist),
                      range(shape[0] / 2 - x_depth_beyond_midline, shape[0]),
                      range(shape[1] / 2 - y_depth_beyond_midline, shape[1])]

    back_left_leg = [range(shape[2] / 2 + height_above_waist),
                     range(shape[0] / 2 + 1 + x_depth_beyond_midline - adj0),
                     range(shape[1] / 2 - y_depth_beyond_midline, shape[1])]

    legs = [front_right_leg, front_left_leg, back_right_leg, back_left_leg]

    if cut_leg is not None:
        deleted_leg = legs[cut_leg]
        deleted_leg[0] = range(double_cut, shape[2] / 2 + height_above_waist - half_cut + double_cut)
        del legs[cut_leg]

    # delete non-leg vox below waist
    for z in range(shape[2] / 2 + height_above_waist):
        for x in range(shape[0]):
            for y in range(shape[1]):
                if bot[x, y, z]:
                    bot[x, y, z] = 0

    # put back legs
    for selected_leg in legs:
        for z in selected_leg[0]:
            for x in selected_leg[1]:
                for y in selected_leg[2]:
                    bot[x, y, z] = mat

    # scar tissue
    if cut_leg is not None:
        for z in deleted_leg[0]:
            for x in deleted_leg[1]:
                for y in deleted_leg[2]:
                    bot[x, y, z] = patch_mat

    if pad > 0:
        bot = np.pad(bot, pad_width=pad, mode='constant', constant_values=0)
        bot = bot[:, :, 1:]

    # put back stump
    if half_cut and cut_leg is not None:
        for z in range(shape[2] / 2 + height_above_waist + pad):
            for x in deleted_leg[1]:
                for y in deleted_leg[2]:
                    if z < max(pad, 1):
                        bot[x + pad, y + pad, z] = patch_mat
                    else:
                        bot[x + pad, y + pad, z] = mat

    return bot
