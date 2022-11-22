#pragma once

#include <CL/sycl.hpp>

#define LOCAL_ACCESSOR_TEMPLATE_PARAMETERS dataT,dimensions,sycl::access::mode::read_write,sycl::access::target::local

template<typename dataT, int dimensions>
 class local_accessor : public sycl::accessor<LOCAL_ACCESSOR_TEMPLATE_PARAMETERS>
{
public:
  // One-dimensional convenience constructor
  template
  <
    typename dataU       = dataT,
    int      dimensionsU = dimensions,
    typename             = std::enable_if_t<1 == dimensionsU>
  >
  local_accessor(size_t s, sycl::handler &h)
    : sycl::accessor<LOCAL_ACCESSOR_TEMPLATE_PARAMETERS>(sycl::range<dimensionsU>(s), h)
  {
    // void
  }

  // N-dimensional ctor
  local_accessor(sycl::range<dimensions> const& r, sycl::handler &h)
    : sycl::accessor<LOCAL_ACCESSOR_TEMPLATE_PARAMETERS>(r, h)
  {
    // void
  }
};
