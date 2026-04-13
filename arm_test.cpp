#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "arm.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("  %s <servo_id> <angle> [time_ms]  - set angle\n", argv[0]);
        printf("  %s grab                           - test grab\n", argv[0]);
        printf("  %s release                        - test release\n", argv[0]);
        printf("  %s release_pos                    - test release position\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s 0 180       - servo 0 to 180 degrees\n", argv[0]);
        printf("  %s 1 90 500    - servo 1 to 90 degrees in 500ms\n", argv[0]);
        printf("  %s grab         - run grab sequence\n", argv[0]);
        return 0;
    }

    Arm arm;

    if (strcmp(argv[1], "grab") == 0) {
        printf("Testing grab...\n");
        arm.grab();
    } else if (strcmp(argv[1], "release") == 0) {
        printf("Testing release...\n");
        arm.release();
    } else if (strcmp(argv[1], "release_pos") == 0) {
        printf("Testing release_pos...\n");
        arm.release_pos();
    } else {
        int servo_id = atoi(argv[1]);
        float angle = atof(argv[2]);
        int time_ms = argc > 3 ? atoi(argv[3]) : 1000;
        printf("Servo %d -> angle %.1f, %dms\n", servo_id, angle, time_ms);
        arm.set_angle(servo_id, angle, time_ms);
    }

    return 0;
}
