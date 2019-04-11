#!/usr/bin/env python
from collections import defaultdict
from numpy.distutils.core import setup
import omr_musicsheet
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List


def build_dict_requirements(
        req_dir: Path = Path('requirements'),
        pattern_req_file: str = '*.pip',
) -> Dict[str, List[Path]]:
    return defaultdict(
        list,
        {
            p.stem: p.read_text().splitlines()
            for p in req_dir.glob(pattern_req_file)
        }
    )


reqs = build_dict_requirements()

descr = """\
"""

DISTNAME = 'omr_musicsheet'
DESCRIPTION = 'OMR MusicSheet with OpenCV'
LONG_DESCRIPTION = descr
MAINTAINER = 'Lionel ATTY',
MAINTAINER_EMAIL = 'yoyonel@hotmail.com',
URL = ''
LICENSE = 'BSD'
DOWNLOAD_URL = ''
PACKAGE_NAME = 'omr_musicsheet'
EXTRA_INFO = dict(
    install_requires=reqs['base'],
    setup_requires=reqs['setup'],
    extras_require={
        'test': reqs['test'],
        'develop': reqs['test'] + reqs['dev'],
        'setup': reqs['test'] + reqs['dev'] + reqs['setup']
    },
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Multimedia :: Audio',
                 'Topic :: Scientific/Engineering'],
    entry_points={
        'console_scripts': [
            'contour_00=omr_musicsheet.contours.contour_00:main'
        ]
    }
)


def configuration(parent_package='', top_path=None, _package_name=PACKAGE_NAME):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg: "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('omr_musicsheet')
    config.add_data_dir('omr_musicsheet/datasets/data')
    return config


# Documentation building command
try:
    from sphinx.setup_command import BuildDoc as SphinxBuildDoc


    class BuildDoc(SphinxBuildDoc):
        """Run in-place build before Sphinx doc build"""

        def run(self):
            ret = subprocess.call(
                [sys.executable, sys.argv[0], 'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building Scipy failed!")
            SphinxBuildDoc.run(self)


    cmdclass = {'build_sphinx': BuildDoc}
except ImportError:
    cmdclass = {}

# Call the setup function
if __name__ == "__main__":
    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          include_package_data=True,
          test_suite="nose.collector",
          cmdclass=cmdclass,
          version=omr_musicsheet.__version__,
          **EXTRA_INFO)
