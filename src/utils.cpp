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


std::string path_stripext(const std::string& str) {
    size_t i = str.find_last_of('.');
    return str.substr(0,i);
}

std::string path_filename(const std::string& str) {
    size_t i = str.find_last_of('/');
    if (i == std::string::npos) i = 0;
    return str.substr(i);
}

// This function creates the full path to a file (creates all the directories)
int mkdir_p(char* file_path_, mode_t mode) {
  char file_path[1024];
  if (file_path_ == NULL) return -1;
  if (*file_path_ == '\0') return 0;
  if (strlen(file_path_) >= STRING_LEN) {
    error("too long path in mkdir: %s\n", file_path_);
    return -1;
  }
  strcpy(file_path,file_path_);
  debug1("mkdir: %s (-p)\n", file_path);
  char* p;
  for (p=strchr(file_path+1, '/'); p; p=strchr(p+1, '/')) {
    *p='\0';
    if (mkdir(file_path, mode)==-1) {
      if (errno!=EEXIST) {
        debug1("mkdir: %s - cannot create\n", file_path);
        *p='/'; return -1;
      } else {
        debug1("mkdir: %s - exists\n", file_path);
      }
    } else {
        debug1("mkdir: %s - created\n", file_path);
    }
    *p='/';
  }
  return 0;
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