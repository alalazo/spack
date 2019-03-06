# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""Schema for env.yaml configuration file.

.. literalinclude:: ../spack/schema/env.py
   :lines: 36-
"""
from llnl.util.lang import union_dicts

import spack.schema.merged
import spack.schema.projections


spec_list_schema = {
    'type': 'array',
    'default': [],
    'items': {
        'anyOf': [
            {'type': 'object',
             'additionalProperties': False,
             'properties': {
                 'matrix': {
                     'type': 'array',
                     'items': {
                         'type': 'array',
                         'items': {
                             'type': 'string',
                         }
                     }
                 },
                 'exclude': {
                     'type': 'array',
                     'items': {
                         'type': 'string'
                     }
                 }
             }
            },
            {'type': 'string'},
            {'type': 'null'}
        ]
    }
}


schema = {
    '$schema': 'http://json-schema.org/schema#',
    'title': 'Spack environment file schema',
    'type': 'object',
    'additionalProperties': False,
    'patternProperties': {
        '^env|spack$': {
            'type': 'object',
            'default': {},
            'additionalProperties': False,
            'properties': union_dicts(
                # merged configuration scope schemas
                spack.schema.merged.properties,
                # extra environment schema properties
                {
                    'include': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        },
                    },
                    'view': {
                        'type': ['boolean', 'string']
                    },
                    'definitions': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'when': {
                                    'type': 'string'
                                }
                            },
                            'patternProperties': {
                                '\w*': spec_list_schema
                            }
                        }
                    },
                    'specs': spec_list_schema,
                    'view': {
                        'anyOf': [
                            {'type': 'boolean'},
                            {'type': 'string'},
                            {'type': 'object',
                             'required': ['root'],
                             'additionalProperties': False,
                             'properties': {
                                 'root': {
                                     'type': 'string'
                                 },
                                 'select': {
                                     'type': 'array',
                                     'items': {
                                         'type': 'string'
                                     }
                                 },
                                 'exclude': {
                                     'type': 'array',
                                     'items': {
                                         'type': 'string'
                                     }
                                 },
                                 'projections': spack.schema.projections.properties['projections']
                             }
                            }
                        ]
                    },
                    'concretize_together': {
                        'type': 'boolean',
                        'default': False
                    }
                }
            )
        }
    }
}
