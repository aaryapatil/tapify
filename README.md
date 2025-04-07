# tapify

[![Logo](https://github.com/aaryapatil/tapify/blob/main/docs/images/tapify_logo.png)](https://github.com/aaryapatil/tapify)

[![Build Status](https://github.com/aaryapatil/tapify/workflows/Tests/badge.svg)](https://github.com/aaryapatil/tapify/actions)
[![License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/aaryapatil/tapify/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/tapify/badge/?version=latest)](http://tapify.readthedocs.io/en/latest/?badge=latest)

A Multitaper Periodogram package for computing the power spectrum of a
time-series with minimal spectral leakage and reduced variance.

# Documentation
Read the documentation at [tapify.readthedocs.io](https://tapify.readthedocs.io/)

The following is a basic use case to get you started:

```
from tapify import MultiTaper

mt_object = MultiTaper(data, t=time, NW=4, K=7)
freq, power = mt_object.periodogram(method='fft',
                                    adaptive_weighting=True,
                                    N_padded='default')
```

# Installation
``pip install tapify``

``python setup.py install``

# Attribution

Please cite `Patil, Eadie, Speagle, Thomson (2022) https://arxiv.org/abs/2209.15027` if you use this code
in your research.
