import vapoursynth as vs
from vsutil import depth
from rekt import rektlvls
from awsmfunc import bbmod
core = vs.core

src = core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][190731][リトル・ニモ Blu-ray]\BFTD_0312\BDMV\STREAM\00003.m2ts")
crop = core.std.CropRel(src, 6, 0, 108, 106)
crop = depth(crop, 10)

borders1 = rektlvls(crop, colnum=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], colval=[20, -10, 6, 2, -3, 3, 3, 2, 3, 2])
borders2 = bbmod(borders1, 0, 0, 10, 0, thresh = 2, blur = 200, u = False, v = False)
borders3 = bbmod(borders2, 0, 0, 10, 0, thresh = 2, blur = 999, y = False)
borders4 = bbmod(borders2, 0, 0, 2, 0, thresh = 6, blur = 999, y = False)



borders4.set_output()

