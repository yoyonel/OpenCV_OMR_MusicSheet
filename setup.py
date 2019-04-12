# !/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from setuptools import setup, find_packages

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

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    include_package_data=True,
    package_data={DISTNAME: ['*.png', '*.jpg']},
    packages=find_packages(exclude='tests'),
    install_requires=reqs['base'],
    use_scm_version=True,
    setup_requires=reqs['setup'],
    extras_require={
        'test': reqs['test'],
        'develop': reqs['test'] + reqs['dev'],
        'setup': reqs['test'] + reqs['dev'] + reqs['setup']
    },
    test_suite='tests',
    tests_require=reqs['test'],
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
            'contour_00=omr_musicsheet.contours.contour_00:main',
            'sprites_sheet=omr_musicsheet.sprites_sheet.aabbox_on_sprites_sheet:main'
        ]
    }
)
