#include "nvpro_pyramid_dispatch_alternative.hpp"

#include "../py2_dispatch_impl.hpp"

NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER(py2_8_16_16,
                                     (py2_dispatch_impl<8, 16, 16>))
