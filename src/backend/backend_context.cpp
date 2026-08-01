#include "backend_context.h"

BackendContext& backend_context() {
    static BackendContext ctx;
    return ctx;
}
