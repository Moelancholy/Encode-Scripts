import vapoursynth as vs
from muvsfunc import SSIM_downsample
from vsdpir import DPIR
from vsutil import get_y, depth
import havsfunc as haf
import lvsfunc as lvf
import vardefunc as vdf
import xvs

from adptvgrnMod import adptvgrnMod

core = vs.core

ep20 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00009.m2ts")[720:3012]
ep21 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00000.m2ts")[720:3012]
ep22 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00001.m2ts")[720:3012]
ep23 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00002.m2ts")[720:3012]
ep24 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00003.m2ts")[720:3012]
ep25 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00004.m2ts")[720:3012]
ep26 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00005.m2ts")[720:3012]
ep27 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00006.m2ts")[720:3012]
ep28 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00007.m2ts")[720:3012]
ep29 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00008.m2ts")[720:3012]
ep30 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV3\BDMV\STREAM\00009.m2ts")[720:3012]
ep31 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV4\BDMV\STREAM\00000.m2ts")[720:3012]
ep32 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV4\BDMV\STREAM\00001.m2ts")[720:3012]
ep33 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV4\BDMV\STREAM\00002.m2ts")[720:3012]
ep34 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV4\BDMV\STREAM\00003.m2ts")[720:3012]
ep35 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV4\BDMV\STREAM\00004.m2ts")[720:3012]
ep36 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV4\BDMV\STREAM\00005.m2ts")[720:3012]

ep20crop = depth(core.std.CropRel(ep20, 240, 240, 0, 0), 16)

ep20 = depth(ep20, 32)
ep21 = depth(ep21, 32)
ep22 = depth(ep22, 32)
ep23 = depth(ep23, 32)
ep24 = depth(ep24, 32)
ep25 = depth(ep25, 32)
ep26 = depth(ep26, 32)
ep27 = depth(ep27, 32)
ep28 = depth(ep28, 32)
ep29 = depth(ep29, 32)
ep30 = depth(ep30, 32)
ep31 = depth(ep31, 32)
ep32 = depth(ep32, 32)
ep33 = depth(ep33, 32)
ep34 = depth(ep34, 32)
ep35 = depth(ep35, 32)
ep36 = depth(ep36, 32)


clips = [ep20, ep21, ep22, ep23, ep24, ep25, ep26, ep27, ep28, ep29, ep30, ep31, ep32, ep33, ep34, ep35, ep36]
average = depth(core.average.Mean(clips), 16)
crop = core.std.CropRel(average, 240, 240, 0, 0)



mask1 = vdf.mask.PrewittStd().get_mask(get_y(crop), multi=2.00).std.Minimum().std.Maximum().std.Maximum().std.Binarize(40<<7).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
mask2 = mask1.rgvs.RemoveGrain(20)
mask1 = mask1.resize.Bicubic(960, 720)
crop = adptvgrnMod(crop, strength=0.45, size=1.30, luma_scaling=1.5, sharp=95, static=False, grain_chroma = False)


merge = core.std.MaskedMerge(ep20crop, crop, mask2)
merge = core.std.ShufflePlanes([merge, crop], [0, 1, 2], vs.YUV)
merge444 = depth(vdf.scale.to_444(merge, 1440, 1080, join_planes = True), 32)
mergeRGB = core.resize.Bicubic(merge444, format = vs.RGBS)

RGB_down_y = SSIM_downsample(mergeRGB, 960, 720, smooth = ((3 ** 2 - 1) / 12) ** 0.5, sigmoid = True, filter_param_a=-1/2, filter_param_b=1/4)
RGB_down_y = get_y(core.resize.Bicubic(RGB_down_y, format = vs.YUV420P16, matrix = 1))
RGB_down_uv = core.resize.Bicubic(mergeRGB, 960, 720)

alt_uv = core.resize.Bicubic(RGB_down_uv, format = vs.YUV420P16, matrix = 1)
merge = core.std.ShufflePlanes([RGB_down_y, alt_uv], [0, 1, 2], vs.YUV)

cden = core.ccd.CCD(RGB_down_uv, 15)
cdenoise = core.std.ShufflePlanes([RGB_down_y, core.resize.Bicubic(cden, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)
cdenoise = lvf.rfs(cdenoise, merge, [(152, 153), (221, 223), 231, 254, (263, 296), (625, 630), (1330, 1332), (1336, 1341), (1348, 1350), (1477, 1522), (1897, 1901), (2043, 2089), (2096, 2097), (2153, 2179)])

dering = haf.EdgeCleaner(cdenoise, strength = 6, rep = True, rmode = 17, smode = 1, hot = True)
dering = adptvgrnMod(dering, strength=0.15, size=1.30, luma_scaling=1, sharp=95, static=False, grain_chroma = False)
dering_m = core.std.MaskedMerge(cdenoise, dering, mask1)

deblock = depth(vdf.scale.to_444(dering_m, 960, 720, join_planes = True), 32)
deblock = core.resize.Bicubic(deblock, format = vs.RGBS)
deblock = DPIR(deblock, strength = 15, task = 'deblock', device_type = 'cpu')
deblock = core.resize.Bicubic(deblock, format = vs.YUV420P16, matrix = 1)
deblock = adptvgrnMod(deblock, strength=0.15, size=1.80, luma_scaling=1, sharp=60, static=False, grain_chroma = False)
luma_mask = core.adg.Mask(dering_m.std.PlaneStats(), luma_scaling=18)
deblock_dark = core.std.MaskedMerge(dering_m, deblock, luma_mask)
deblock_final = lvf.rfs(dering_m, deblock_dark, [(83, 90), (123, 130), (148, 151), (228, 230), (237, 238), (631, 645), (1144, 1171), (1313, 1330), (1339, 1342), (1817, 1820), (1876, 1892), (1897, 1901), (2189, 2210)])


out = depth(deblock_final, 10)

out.set_output()
#merge.set_output(2)
#cdenoise.set_output(3)
#ep34.set_output(6)
#ep35.set_output(7)
#ep36.set_output(8)
#ep37.set_output(9)


###GOTTA SCENEFILTER CDEN