#include "tasks.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

const TaskInfo TASKS[T_COUNT] = {
    {"add", 5}, {"sub", 5}, {"mul", 4}, {"cmp", 5},
    {"rev", 8}, {"sort", 8}, {"count", 10},
};

static long long ipow10(int k) {
    long long r = 1;
    while (k-- > 0) r *= 10;
    return r;
}

static long long rand_digits(int nd) {
    long long lo = nd <= 1 ? 0 : ipow10(nd - 1), hi = ipow10(nd) - 1;
    return lo + (long long)(rng_u64() % (unsigned long long)(hi - lo + 1));
}

static void reverse(char *s) {
    for (size_t i = 0, j = strlen(s); i + 1 < j; i++, j--) {
        char t = s[i]; s[i] = s[j - 1]; s[j - 1] = t;
    }
}

static void rand_str(char *s, int len, int alphabet) {
    for (int i = 0; i < len; i++) s[i] = (char)('a' + rng_int(alphabet));
    s[len] = '\0';
}

void task_gen(int task, int level, char *prompt, size_t cap) {
    if (level < 1) level = 1;
    long long a = rand_digits(level), b = rand_digits(1 + rng_int(level));
    if (rng_int(2)) { long long t = a; a = b; b = t; }
    char s[64];
    switch (task) {
    case T_ADD: snprintf(prompt, cap, "%lld+%lld=", a, b); break;
    case T_SUB:
        if (b > a) { long long t = a; a = b; b = t; }
        snprintf(prompt, cap, "%lld-%lld=", a, b);
        break;
    case T_MUL:
        snprintf(prompt, cap, "%lld*%d=", rand_digits(level), rng_int(10));
        break;
    case T_CMP:
        a = rand_digits(level);
        b = rng_int(10) == 0 ? a : rand_digits(level);
        if (rng_int(2) && level > 1) { /* share a prefix: harder */
            long long m = ipow10(rng_int(level - 1) + 1);
            b = a / m * m + (long long)(rng_u64() % (unsigned long long)m);
        }
        snprintf(prompt, cap, "%lld?%lld=", a, b);
        break;
    case T_REV: rand_str(s, level, 26); snprintf(prompt, cap, "r:%s=", s); break;
    case T_SORT: rand_str(s, level, 10); snprintf(prompt, cap, "s:%s=", s); break;
    default: rand_str(s, level, 3); snprintf(prompt, cap, "c:%s=", s); break;
    }
}

int task_of(const char *p) {
    if (p[0] == 'r' && p[1] == ':') return T_REV;
    if (p[0] == 's' && p[1] == ':') return T_SORT;
    if (p[0] == 'c' && p[1] == ':') return T_CNT;
    if (strchr(p, '+')) return T_ADD;
    if (strchr(p, '*')) return T_MUL;
    if (strchr(p, '?')) return T_CMP;
    if (strchr(p, '-')) return T_SUB;
    return -1;
}

static int cmp_char(const void *x, const void *y) {
    return *(const char *)x - *(const char *)y;
}

int task_solve(const char *prompt, char *ans, size_t cap) {
    int t = task_of(prompt);
    char body[64];
    long long a, b;
    char op;
    switch (t) {
    case T_ADD: case T_SUB: case T_MUL: case T_CMP:
        if (sscanf(prompt, "%lld%c%lld=", &a, &op, &b) != 3) return -1;
        if (t == T_CMP) {
            snprintf(ans, cap, "%c", a < b ? '<' : a > b ? '>' : '=');
        } else {
            long long r = t == T_ADD ? a + b : t == T_SUB ? a - b : a * b;
            snprintf(ans, cap, "%lld", r);
            reverse(ans);
        }
        return 0;
    case T_REV: case T_SORT: case T_CNT: {
        const char *eq = strchr(prompt, '=');
        size_t n = eq ? (size_t)(eq - prompt - 2) : 0;
        if (!eq || n >= sizeof(body)) return -1;
        memcpy(body, prompt + 2, n);
        body[n] = '\0';
        if (t == T_REV) reverse(body);
        if (t == T_SORT) qsort(body, n, 1, cmp_char);
        if (t == T_CNT) {
            int k = 0;
            for (size_t i = 0; i < n; i++) k += body[i] == 'a';
            snprintf(body, sizeof(body), "%d", k);
        }
        snprintf(ans, cap, "%s", body);
        return 0;
    }
    default: return -1;
    }
}

int task_check(const char *prompt, const char *answer) {
    char ref[64];
    if (task_solve(prompt, ref, sizeof(ref)) != 0) return 0;
    return strcmp(ref, answer) == 0;
}

void task_pretty(int task, const char *answer, char *out, size_t cap) {
    snprintf(out, cap, "%s", answer);
    if (task == T_ADD || task == T_SUB || task == T_MUL) reverse(out);
}

int task_parse(const char *in, char *prompt, size_t cap) {
    char buf[128];
    size_t k = 0;
    while (*in == ' ') in++;
    const char *words[] = {"rev ", "sort ", "count "};
    const char *tags[] = {"r:", "s:", "c:"};
    for (int i = 0; i < 3; i++) {
        size_t wl = strlen(words[i]);
        if (strncmp(in, words[i], wl) == 0) {
            snprintf(prompt, cap, "%s%s=", tags[i], in + wl);
            return task_of(prompt);
        }
    }
    int is_cmp = strncmp(in, "cmp ", 4) == 0;
    if (is_cmp) in += 4;
    for (; *in && k + 2 < sizeof(buf); in++) {
        if (*in == ' ') {
            if (is_cmp && k && buf[k - 1] != '?') buf[k++] = '?';
            continue;
        }
        buf[k++] = *in == 'x' ? '*' : *in;
    }
    buf[k] = '\0';
    if (k && buf[k - 1] == '=') buf[--k] = '\0';
    for (size_t i = 0; i < k; i++)
        if (!isdigit((unsigned char)buf[i]) && !strchr("+-*?", buf[i]))
            return -1;
    snprintf(prompt, cap, "%s=", buf);
    return task_of(prompt);
}
