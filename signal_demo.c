#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>

static volatile sig_atomic_t g_running = 1;

static void signal_handler(int sig) {
    printf("=== signal %d received, g_running=0 ===\n", sig);
    fflush(stdout);
    g_running = 0;
}

int main() {
    printf("PID=%d\n", getpid());
    fflush(stdout);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("=== signals registered, entering while loop ===\n");
    fflush(stdout);

    while (g_running) {
        printf(".");
        fflush(stdout);
        sleep(1);
    }

    printf("\n=== exiting ===\n");
    return 0;
}
