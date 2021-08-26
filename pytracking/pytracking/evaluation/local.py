from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = r'E:\Mare5\dev\projects\diploma-tracking\datasets\DAVIS'
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = r'E:\Mare5\dev\projects\diploma-tracking\pytracking\pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = r'E:\Mare5\dev\projects\diploma-tracking\datasets\otb'
    settings.result_plot_path = r'E:\Mare5\dev\projects\diploma-tracking\pytracking\pytracking/result_plots/'
    settings.results_path = r'E:\Mare5\dev\projects\diploma-tracking\pytracking\pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = r'E:\Mare5\dev\projects\diploma-tracking\pytracking\pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = r'E:\Mare5\dev\projects\diploma-tracking\datasets\vot2020\sequences'
    settings.youtubevos_dir = ''

    settings.synthetic_path = r'E:\Mare5\dev\projects\diploma-tracking\datasets\synthetic/'

    return settings

