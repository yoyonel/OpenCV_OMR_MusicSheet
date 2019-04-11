def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('omr_musicsheet', parent_package, top_path)

    config.add_subpackage('sprites_sheet')
    config.add_subpackage('contours')
    config.add_subpackage('datasets')
    config.add_subpackage('histogram')
    config.add_subpackage('LineSegmentDetector')
    config.add_subpackage('texture_synthesis')
    config.add_subpackage('tools')
    config.add_subpackage('scikit')

    return config
