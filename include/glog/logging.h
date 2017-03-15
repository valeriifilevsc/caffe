#ifndef _LOGGING_H_
#define _LOGGING_H_

#include <sys/types.h>
#include <unistd.h>
#include <iostream>  // NOLINT

enum LogSeverity {INFO = -1, WARNING = -2, ERROR = -3, FATAL = -4};

#define CHECK_NOTNULL(x) google::CheckNotNull(x)
#define CHECK(x) \
  if (!(x)) std::cerr << "Check failed: " #x
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_GT(x, y) CHECK((x) > (y))

#ifndef NDEBUG
#define LOG(severity) std::cerr
#define DLOG(severity) LOG(severity)
#define DCHECK(x) CHECK(x)
#define DCHECK_EQ(x, y) CHECK_EQ(x, y)
#define DCHECK_NE(x, y) CHECK_NE(x, y)
#define DCHECK_LE(x, y) CHECK_LE(x, y)
#define DCHECK_LT(x, y) CHECK_LT(x, y)
#define DCHECK_GE(x, y) CHECK_GE(x, y)
#define DCHECK_GT(x, y) CHECK_GT(x, y)
#else  // NDEBUG
#define LOG(severity) \
  while (false) \
    std::cerr
#define DLOG(severity) \
  while (false) \
    LOG(severity)
#define DCHECK(condition) \
  while (false) \
    CHECK(condition)
#define DCHECK_EQ(val1, val2) \
  while (false) \
    CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) \
  while (false) \
    CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) \
  while (false) \
    CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) \
  while (false) \
    CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) \
  while (false) \
    CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) \
  while (false) \
    CHECK_GT(val1, val2)
#define DCHECK_STREQ(str1, str2) \
  while (false) \
    CHECK_STREQ(str1, str2)
#endif

namespace google {
    void inline InitGoogleLogging(const char* argv0) {

    }

    void inline InstallFailureSignalHandler() {

    }

    template <typename T>
    T* CheckNotNull(T* t) {
        if (t == NULL) {
            throw 1;
        }
        return t;
    }
}

#endif  // _LOGGING_H_
