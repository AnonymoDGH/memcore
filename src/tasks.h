#ifndef MC_TASKS_H
#define MC_TASKS_H

#include <stddef.h>

/* Verifiable synthetic tasks. Each example is "<prompt><answer>\n".
   Arithmetic answers are written least-significant digit first so the
   model emits digits in carry order (Lee et al., 2023). */
enum { T_ADD, T_SUB, T_MUL, T_CMP, T_REV, T_SORT, T_CNT, T_COUNT };

typedef struct {
    const char *name;
    int train_max; /* curriculum ceiling; higher levels test generalization */
} TaskInfo;

extern const TaskInfo TASKS[T_COUNT];

void task_gen(int task, int level, char *prompt, size_t cap);
/* reference solver: the verifier. Returns 0 on success. */
int task_solve(const char *prompt, char *ans, size_t cap);
int task_check(const char *prompt, const char *answer);
int task_of(const char *prompt);
/* human-readable answer (un-reverses digits) */
void task_pretty(int task, const char *answer, char *out, size_t cap);
/* "123+45", "rev hello", "sort dcba", "cmp 12 34", "count abca" */
int task_parse(const char *in, char *prompt, size_t cap);

#endif
