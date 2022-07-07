from src.GP_Bald import bald_acquisition
from src.GP_Bald_WS import bald_ws_acquisition
from src.GP_MI_WS import mi_ws_acquisition
from src.GP_Bald_WS_mat import bald_ws_acquisition as bald_ws_acquisition_mat
from src.GP_MI_WS_mat import mi_ws_acquisition as mi_ws_acquisition_mat
from src.GP_diff_WS import diff_ws_acquisition
from src.GPC_Bald_WS import gpc_bald_ws_acquisition
from src.GPC_Bald_WS_samp import gpc_bald_ws_samp_acquisition
from src.GP_Bald_WS_fl import bald_ws_fl_acquisition
from src.GP_MI_WS_fl import mi_ws_fl_acquisition

def can_handle_noisy_annotations(acq_fun):
    can_handle = [bald_ws_acquisition, mi_ws_acquisition, diff_ws_acquisition, gpc_bald_ws_acquisition,
                  gpc_bald_ws_samp_acquisition, bald_ws_acquisition_mat, mi_ws_acquisition_mat]

    return acq_fun in can_handle


def can_handle_noisy_annotations_test():

    assert not can_handle_noisy_annotations(bald_acquisition)
    assert can_handle_noisy_annotations(bald_ws_acquisition)
    assert can_handle_noisy_annotations(mi_ws_acquisition)
    assert can_handle_noisy_annotations(diff_ws_acquisition)
    assert can_handle_noisy_annotations(gpc_bald_ws_acquisition)
    assert can_handle_noisy_annotations(gpc_bald_ws_samp_acquisition)


can_handle_noisy_annotations_test()
