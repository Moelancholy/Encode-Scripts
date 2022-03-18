import vapoursynth as vs
from vsdpir import DPIR
from vsutil import get_y, depth
import havsfunc as haf
import lvsfunc as lvf
import vardefunc as vdf
import xvs

from adptvgrnMod import adptvgrnMod

core = vs.core

ep01 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00000.m2ts")[:2832]
ep02 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00001.m2ts")[:2832]
ep03 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00002.m2ts")[:2832]
ep04 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00003.m2ts")[:2832]
ep05 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00004.m2ts")[:2832]
ep06 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00005.m2ts")[:2832]
ep07 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00006.m2ts")[:2832]
ep08 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00007.m2ts")[:2832]
ep09 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00008.m2ts")[:2832]
ep10 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV1\BDMV\STREAM\00009.m2ts")[:2832]
ep11 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00000.m2ts")[:2832]
ep12 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00001.m2ts")[:2832]
ep13 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00002.m2ts")[:2832]
ep14 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00003.m2ts")[:2832]
ep15 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00004.m2ts")[:2832]
ep16 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00005.m2ts")[:2832]
ep17 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00006.m2ts")[:2832]
ep18 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00007.m2ts")[:2832]
ep19 = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV2\BDMV\STREAM\00008.m2ts")[:2832]
ep01crop = depth(core.std.CropRel(ep01, 240, 240, 0, 0), 16)

ep01 = depth(ep01, 32)
ep02 = depth(ep02, 32)
ep03 = depth(ep03, 32)
ep04 = depth(ep04, 32)
ep05 = depth(ep05, 32)
ep06 = depth(ep06, 32)
ep07 = depth(ep07, 32)
ep08 = depth(ep08, 32)
ep09 = depth(ep09, 32)
ep10 = depth(ep10, 32)
ep11 = depth(ep11, 32)
ep12 = depth(ep12, 32)
ep13 = depth(ep13, 32)
ep14 = depth(ep14, 32)
ep15 = depth(ep15, 32)
ep16 = depth(ep16, 32)
ep17 = depth(ep17, 32)
ep18 = depth(ep18, 32)
ep19 = depth(ep19, 32)


clips = [ep01, ep02, ep03, ep04, ep05, ep06, ep07, ep08, ep09, ep10, ep11, ep12, ep13, ep14, ep15, ep16, ep17, ep18, ep19]
average = depth(core.average.Mean(clips), 16)
crop = core.std.CropRel(average, 240, 240, 0, 0)
crop = adptvgrnMod(crop, strength=0.45, size=1.30, luma_scaling=1, sharp=95, static=False, grain_chroma = False)
mask1 = vdf.mask.PrewittStd().get_mask(get_y(crop), multi=2.00).std.Minimum().std.Maximum().std.Maximum().std.Binarize(40<<7).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
mask2 = vdf.mask.PrewittStd().get_mask(get_y(crop), multi=2.00).std.Minimum().std.Maximum().std.Binarize(50<<7).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)

merge = core.std.MaskedMerge(ep01crop, crop, mask2)
merge = core.std.ShufflePlanes([merge, crop], [0, 1, 2], vs.YUV)
c_lines = xvs.mwcfix(merge, kernel = 1, restore = 5, a = 2, grad = 4, warp = 6, thresh = 128, blur = 3, repair = 1)
c_lines_m = core.std.MaskedMerge(merge, c_lines, mask1)
c_lines_m = lvf.rfs(c_lines_m, merge, [(1415, 1423), (1433, 1471)])


crest444 = depth(vdf.scale.to_444(c_lines_m, 1440, 1080, join_planes = True), 32)
crestRGB = core.resize.Bicubic(crest444, format = vs.RGBS)
dpir1 = DPIR(crestRGB, strength = 10, task = 'deblock', device_type = 'cpu')
dpir2 = DPIR(crestRGB, strength = 20, task = 'deblock', device_type = 'cpu')
dpir3 = DPIR(crestRGB, strength = 30, task = 'deblock', device_type = 'cpu')

cden = core.ccd.CCD(crestRGB, 20)
cdenoise = core.std.ShufflePlanes([c_lines_m, core.resize.Bicubic(cden, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)

dering = haf.EdgeCleaner(cdenoise, strength = 9, rep = True, rmode = 17, smode = 1, hot = True)
dering = adptvgrnMod(dering, strength=0.15, size=1.30, luma_scaling=1, sharp=95, static=False, grain_chroma = False)
dering_m = core.std.MaskedMerge(cdenoise, dering, mask1)

deblock = depth(vdf.scale.to_444(dering_m, 1440, 1080, join_planes = True), 32)
deblock = core.resize.Bicubic(deblock, format = vs.RGBS)
deblock = DPIR(deblock, strength = 30, task = 'deblock', device_type = 'cpu')
deblock = core.resize.Bicubic(deblock, format = vs.YUV420P16, matrix = 1)
deblock = adptvgrnMod(deblock, strength=0.15, size=1.30, luma_scaling=1, sharp=90, static=False, grain_chroma = False)
luma_mask = core.adg.Mask(dering_m.std.PlaneStats(), luma_scaling=18)
deblock_dark = core.std.MaskedMerge(dering_m, deblock, luma_mask)
deblock_final = lvf.rfs(dering_m, deblock_dark, [(360, 385), (699, 707), (726, 734), (1472, 1480), (1618, 1626), (1710, 1729), (1939, 1946), (2324, 2332), (2342, 2350)])


out = depth(deblock_final, 10)

out.set_output()
#dering_m.set_output(2)
#deblock_dark.set_output(3)
#dpir3.set_output(4)
