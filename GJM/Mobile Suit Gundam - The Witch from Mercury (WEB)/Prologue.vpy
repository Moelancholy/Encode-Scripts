import vapoursynth as vs


core = vs.core


import debandshit as dbs
import lvsfunc as lvf

from adptvgrnMod import adptvgrnMod
from kagefunc import retinex_edgemask
from vardefunc.scale import to_444
from vsdpir import DPIR
from vsutil import get_depth, get_y, depth, plane


###CUSTOM FUNCS
def merge_tvcr(tv, cr, radius=3, passes=2, lehmer=True):
    tv_high = core.std.MakeDiff(tv, tv.std.BoxBlur(hradius=radius, hpasses=passes, vradius=radius, vpasses=passes))
    tv_low = core.std.MakeDiff(tv, tv_high)
    cr_high = core.std.MakeDiff(cr, cr.std.BoxBlur(hradius=radius, hpasses=passes, vradius=radius, vpasses=passes))
    if lehmer:
        cr_hi = core.std.Expr([tv_high, cr_high], "x 32768 - 3 pow y 32768 - 3 pow + x 32768 - 2 pow y 32768 - 2 pow 0.0001 + + / 32768 +")
    return core.std.MergeDiff(tv_low, cr_hi)


def ccdmod(src: vs.VideoNode, threshold: float = 4, matrix: int = 1) -> vs.VideoNode:
    """
    A faster implementation that processes only chroma. Taken from jvsfunc.
    """

    if src.format.color_family != vs.YUV:
        raise ValueError('ccdmod: only YUV format is supported, use jvsfunc.ccd() for RGB.')

    if matrix is None:
        matrix = 1 if src.width > 1270 or src.height > 710 else 6

    thr = threshold**2/195075.0
    u = plane(src, 1)

    yuv = src.resize.Bicubic(u.width, u.height, format=vs.YUV444P16)
    rgbs = yuv.resize.Point(format=vs.RGBS, matrix_in=matrix)

    rrr = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [0, 0, 0], vs.RGB)
    ggg = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [1, 1, 1], vs.RGB)
    bbb = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [2, 2, 2], vs.RGB)

    ex_ccd = core.akarin.Expr([yuv, rrr, ggg, bbb],
                              ['',
                               'y[-12,-12] y - 2 pow z[-12,-12] z - 2 pow a[-12,-12] a - 2 pow + + A! '
                               'y[-4,-12] y - 2 pow z[-4,-12] z - 2 pow a[-4,-12] a - 2 pow + + B! '
                               'y[4,-12] y - 2 pow z[4,-12] z - 2 pow a[4,-12] a - 2 pow + + C! '
                               'y[12,-12] y - 2 pow z[12,-12] z - 2 pow a[12,-12] a - 2 pow + + D! '
                               'y[-12,-4] y - 2 pow z[-12,-4] z - 2 pow a[-12,-4] a - 2 pow + + E! '
                               'y[-4,-4] y - 2 pow z[-4,-4] z - 2 pow a[-4,-4] a - 2 pow + + F! '
                               'y[4,-4] y - 2 pow z[4,-4] z - 2 pow a[4,-4] a - 2 pow + + G! '
                               'y[12,-4] y - 2 pow z[12,-4] z - 2 pow a[12,-4] a - 2 pow + + H! '
                               'y[-12,4] y - 2 pow z[-12,4] z - 2 pow a[-12,4] a - 2 pow + + I! '
                               'y[-4,4] y - 2 pow z[-4,4] z - 2 pow a[-4,4] a - 2 pow + + J! '
                               'y[4,4] y - 2 pow z[4,4] z - 2 pow a[4,4] a - 2 pow + + K! '
                               'y[12,4] y - 2 pow z[12,4] z - 2 pow a[12,4] a - 2 pow + + L! '
                               'y[-12,12] y - 2 pow z[-12,12] z - 2 pow a[-12,12] a - 2 pow + + M! '
                               'y[-4,12] y - 2 pow z[-4,12] z - 2 pow a[-4,12] a - 2 pow + + N! '
                               'y[4,12] y - 2 pow z[4,12] z - 2 pow a[4,12] a - 2 pow + + O! '
                               'y[12,12] y - 2 pow z[12,12] z - 2 pow a[12,12] a - 2 pow + + P! '
                               f'A@ {thr} < 1 0 ? B@ {thr} < 1 0 ? C@ {thr} < 1 0 ? D@ {thr} < 1 0 ? '
                               f'E@ {thr} < 1 0 ? F@ {thr} < 1 0 ? G@ {thr} < 1 0 ? H@ {thr} < 1 0 ? '
                               f'I@ {thr} < 1 0 ? J@ {thr} < 1 0 ? K@ {thr} < 1 0 ? L@ {thr} < 1 0 ? '
                               f'M@ {thr} < 1 0 ? N@ {thr} < 1 0 ? O@ {thr} < 1 0 ? P@ {thr} < 1 0 ? '
                               '+ + + + + + + + + + + + + + + 1 + Q! '
                               f'A@ {thr} < x[-12,-12] 0 ? B@ {thr} < x[-4,-12] 0 ? '
                               f'C@ {thr} < x[4,-12] 0 ? D@ {thr} < x[12,-12] 0 ? '
                               f'E@ {thr} < x[-12,-4] 0 ? F@ {thr} < x[-4,-4] 0 ? '
                               f'G@ {thr} < x[4,-4] 0 ? H@ {thr} < x[12,-4] 0 ? '
                               f'I@ {thr} < x[-12,4] 0 ? J@ {thr} < x[-4,4] 0 ? '
                               f'K@ {thr} < x[4,4] 0 ? L@ {thr} < x[12,4] 0 ? '
                               f'M@ {thr} < x[-12,12] 0 ? N@ {thr} < x[-4,12] 0 ? '
                               f'O@ {thr} < x[4,12] 0 ? P@ {thr} < x[12,12] 0 ? '
                               '+ + + + + + + + + + + + + + + x + Q@ /'])

    ex_ccd = ex_ccd.resize.Bicubic(src_left=-0.25)
    ex_ccd = depth(ex_ccd, get_depth(src))
    return core.std.ShufflePlanes([src, ex_ccd], [0, 1, 2], vs.YUV)


###SOURCES
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 01 (Amazon dAnime CBR 1080p).mkv")
cbr16 = depth(cbr, 16)
vbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 01 (Amazon dAnime VBR 1080p).mkv")


###DEBLOCK
deblock1 = depth(to_444(cbr, 1920, 1080, join_planes = True), 32)
deblock1 = core.resize.Bicubic(deblock1, format = vs.RGBS)
deblock1 = DPIR(deblock1, strength = 18, task = 'deblock', device_type = 'cpu')
deblock1 = core.resize.Bicubic(deblock1, format = vs.YUV420P16, matrix = 1)

deblock2 = depth(to_444(vbr, 1920, 1080, join_planes = True), 32)
deblock2 = core.resize.Bicubic(deblock2, format = vs.RGBS)
deblock2 = DPIR(deblock2, strength = 18, task = 'deblock', device_type = 'cpu')
deblock2 = core.resize.Bicubic(deblock2, format = vs.YUV420P16, matrix = 1)

merge = merge_tvcr(deblock2, deblock1)


##AA
merge_y = get_y(merge)
aa_mask = merge_y.std.Prewitt().std.Binarize(60 << 8).std.Maximum().std.Convolution([1]*9)

aa1 = lvf.aa.upscaled_sraa(merge_y)
aa2 = lvf.aa.taa(merge_y, lvf.aa.nnedi3(opencl=True))
aa = lvf.aa.clamp_aa(merge_y, aa2, aa1, 1.5)

aa_masked = core.std.MaskedMerge(merge_y, aa, aa_mask)


###CHROMA
aa_merge = core.std.ShufflePlanes([aa_masked, merge], [0, 1, 2], vs.YUV)
chromadn = ccdmod(aa_merge, 4)

linemask = retinex_edgemask(aa_masked)
dn_merged = core.std.MaskedMerge(chromadn, aa_merge, linemask)


###DEBAND
deband = dbs.placebo_deband(dn_merged, 16, 3.5, 3, 0)
deband = core.std.MaskedMerge(deband, dn_merged, linemask)


###OUTPUT
grain = adptvgrnMod(deband, strength = 0.03, static = True, luma_scaling = 2, grain_chroma = False, size = 1.2, sharp = 75)
grain = adptvgrnMod(grain, strength = 0.11, static = False, luma_scaling = 10, grain_chroma = False, size = 1.2, sharp = 75)
out = depth(grain, 10)

out.set_output()