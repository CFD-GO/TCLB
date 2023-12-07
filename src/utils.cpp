#include "utils.h"

static constexpr std::string_view base64char = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void Base64char3(unsigned char* in, int len, char* out) {
    if (len > 3) len = 3;
    int i;
    unsigned int w = 0;
    for (i = 0; i < len; i++) w = (w << 8) + (int)(in[i]);
    for (; i < 3; i++) w = w << 8;
    for (i = 0; i < 4; i++) {
        out[3 - i] = base64char[w & 0x3F];
        w = w >> 6;
    }
    for (i = len; i < 3; i++) out[i + 1] = '=';
}

void fprintB64(FILE* f, const void* tab, size_t len) {
    constexpr size_t B64LineLen = 76;
    char buf[B64LineLen + 1];
    buf[B64LineLen] = 0;
    size_t i, k = 0;
    for (i = 0; i < len;) {
        Base64char3(((unsigned char*)tab) + i, len - i, buf + k);
        i += 3;
        k += 4;
        if (k >= B64LineLen) {
            fprintf(f, "%s", buf);
            k = 0;
        }
    }
    buf[k] = 0;
    fprintf(f, "%s", buf);
}

int mkdir_p(const std::string& path, std::filesystem::perms perms) {
    try {
        const auto fs_path = std::filesystem::path(path).parent_path();
        std::filesystem::create_directories(fs_path);
        std::filesystem::permissions(fs_path, perms);
        return EXIT_SUCCESS;
    } catch (...) {
        debug1("Failed to create %s\n", path.c_str());
        return EXIT_FAILURE;
    }
}

FILE* fopen_gz(const char* filename, const char* mode) {
    bool gzip = false;
    int len = strlen(filename);
    if (len > 3) {
        if (strcmp(&filename[len - 3], ".gz") == 0) { gzip = true; }
    }
    if (gzip) {
        warning("Opening a gzip file: %s (%s)\n", filename, mode);
        if (strcmp(mode, "r") == 0) {
            char cmd[STRING_LEN * 2];
            if (access(filename, R_OK)) return NULL;
            sprintf(cmd, "gzip -d <%s", filename);
            return popen(cmd, "r");
        } else if (strcmp(mode, "rb") == 0) {
            char cmd[STRING_LEN * 2];
            if (access(filename, R_OK)) return NULL;
            sprintf(cmd, "gzip -d <%s", filename);
            return popen(cmd, "r");
        } else if (strcmp(mode, "w") == 0) {
            char cmd[STRING_LEN * 2];
            sprintf(cmd, "gzip >%s", filename);
            return popen(cmd, "w");
        } else if (strcmp(mode, "a") == 0) {
            char cmd[STRING_LEN * 2];
            sprintf(cmd, "gzip >>%s", filename);
            return popen(cmd, "w");
        } else {
            ERROR("Unknown mode for gzip file: fopen_gz('%s','%s')\n", filename, mode);
            return NULL;
        }
    } else {
        return fopen(filename, mode);
    }
    return NULL;
}