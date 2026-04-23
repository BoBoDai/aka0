#ifndef ARM_HPP
#define ARM_HPP

#include <string>

class Arm {
  public:
    Arm(const std::string& port = "/dev/ttyS2", int baudrate = 115200);
    explicit Arm(const std::string& port, int baudrate, const std::string& config_path);
    ~Arm();

    void set_angle(int servo_id, float angle, int time_ms = 1000);
    void release_torque(int servo_id = 255);
    void restore_torque(int servo_id = 255);

    void grab();
    void release();
    void grab_pos();  // 收回到待抓取位置（home）

  private:
    void open_serial(const std::string& port, int baudrate);
    void close_serial();
    void send_command(const std::string& cmd);
    int angle_to_pulse(float angle) const;
    void load_config(const std::string& config_path);
    int parse_int_from_json(const char* json_str, const char* key);

    int fd_;

    static const int PULSE_MIN = 500;
    static const int PULSE_MAX = 2500;
    static const float ANGLE_MAX;

    // Servo angles (loaded from JSON config)
    float SERVO0_READY;
    float SERVO1_READY;
    float SERVO0_LIFT;
    float SERVO1_LIFT;

    // Servo2 specific angles
    float SERVO2_PREPARE;
    float SERVO2_APPROACH;
    float SERVO2_GRAB;
    float SERVO2_LIFT;
};

#endif // ARM_HPP
