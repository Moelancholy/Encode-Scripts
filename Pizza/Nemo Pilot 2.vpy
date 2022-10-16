import vapoursynth as vs
from vsutil import depth
core = vs.core

src = core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][190731][リトル・ニモ Blu-ray]\BFTD_0312\BDMV\STREAM\00002.m2ts").resize.Bicubic(src_top=1)
crop = core.std.CropRel(src, 0, 0, 20, 22)
crop = depth(crop, 10)


crop.set_output()
