////////////////////////////////////////////////////////////////////////////////
///
/// @file basic.h
/// @brief Extremely basic debug logging functionality.
/// @author XeN
/// @author Zenon Parker
/// @date 2014
///
////////////////////////////////////////////////////////////////////////////////

#ifndef XEN_LOG_BASIC_H
#define XEN_LOG_BASIC_H

#ifdef VERBOSE
#define DEBUG_LOG(X) std::cout << X << std::endl;
#else
#define DEBUG_LOG(X) ;
#endif // VERBOSE

namespace xen {
namespace log { 

} // end namespace log
} // end namespace xen

#endif // XEN_LOG_BASIC_H

