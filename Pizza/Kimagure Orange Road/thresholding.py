import vapoursynth as vs
from typing import List, NamedTuple, Optional, Sequence
core = vs.core

class Thresholds(NamedTuple):
    """
    [soft_bound_min, [hard_bound_min, hard_bound_max], soft_bound_max)
    """
    clip: vs.VideoNode
    soft_bound_min: Sequence[float]
    hard_bound_min: Sequence[float]
    hard_bound_max: Sequence[float]
    soft_bound_max: Sequence[float]
    coef_min: None
    coef_max: None

def thresholding(*thrs: Thresholds, base: Optional[vs.VideoNode] = None, guidance: Optional[vs.VideoNode] = None) -> vs.VideoNode:
    """
    TAKEN FROM VARDE THANK YOU MUCH LOVE
    """
    if not base:
        base = thrs[0].clip.std.BlankClip()
    if not guidance:
        guidance = thrs[0].clip

    if not base.format or not guidance.format:
        raise ValueError('thresholding: variable format not allowed')

    for i, thr in enumerate(thrs):
        if thr.clip.format != base.format:
            raise ValueError(f'thresholding: threshold {i} has a different format than base clip')

    def _normalise_thr(thr: Sequence[float], num_planes: int) -> List[float]:
        thr = [thr] if isinstance(thr, (float, int)) else thr
        return (list(thr) + [thr[-1]] * (num_planes - len(thr)))[:num_planes]

    pclip = base

    for thr in thrs:
        soft_bound_min, hard_bound_min, hard_bound_max, soft_bound_max = (_normalise_thr(t, base.format.num_planes) for t in thr[1:5])
        coef_min = _normalise_thr(thr.coef_min, base.format.num_planes) if thr.coef_min else None
        coef_max = _normalise_thr(thr.coef_max, base.format.num_planes) if thr.coef_max else None

        exprs: List[str] = []
        for i in range(base.format.num_planes):
            if_in_min = f'x {soft_bound_min[i]} >= x {hard_bound_min[i]} < and'
            if_in_max = f'x {hard_bound_max[i]} >= x {soft_bound_max[i]} < and'
            if_in_hard = f'x {hard_bound_min[i]} >= x {hard_bound_max[i]} < and'

            str_min = f'x {soft_bound_min[i]} - {hard_bound_min[i]} {soft_bound_min[i]} - /'
            if coef_min:
                str_min += f' {coef_min[i]} pow'

            str_max = f'x {hard_bound_max[i]} - {soft_bound_max[i]} {hard_bound_max[i]} - /'
            if coef_max:
                str_max += f' {coef_max} pow'

            exprs.append(
                if_in_min + f' z {str_min} * y 1 {str_min} - * + '
                + if_in_max + f' y {str_max} * z 1 {str_max} - * + '
                + if_in_hard + ' z y ? ? ?'
            )

        pclip = core.std.Expr([guidance, pclip, thr.clip], exprs)

    return pclip

def kimagure_thresholding(base, cden1, cden2):
    a = thresholding(Thresholds(cden2, [0, 0, 0], [0, 0, 0], [0.65, 1, 1], [0.72, 1, 1], None, None), 
                    Thresholds(cden1, [0.65, 1, 1], [0.72, 1, 1], [1, 1, 1], [1, 1, 1], None, None), base = base)
    return a