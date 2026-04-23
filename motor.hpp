#ifndef MOTOR_HPP
#define MOTOR_HPP

#include <string>

class Motor {
  public:
    Motor();
    explicit Motor(const std::string& config_path);
    ~Motor();

    void forward(int speed);
    void backward(int speed);
    void left(int speed);
    void right(int speed);
    void brake();
    void standby();
    // 差速驱动：正值前进，负值后退，范围 [-100, 100]
    void drive(int left_speed, int right_speed);

  private:
    void set_pwm_duty_cycle(int pwm_id, int duty_cycle);
    void set_pwm_enable(int pwm_id, bool enable);
    void set_speed(int pwm_id, int speed);
    void init_pwm(int pwm_id);
    void load_config(const std::string& config_path);
    int parse_int_from_json(const char* json_str, const char* key);

    const std::string PWM_PATH = "/sys/class/pwm/pwmchip4/";
    const int PERIOD = 10000; // 10kHz

    // PWM IDs for left and right wheels (loaded from JSON config)
    int LEFT_WHEEL_BACKWARD;
    int LEFT_WHEEL_FORWARD;
    int RIGHT_WHEEL_BACKWARD;
    int RIGHT_WHEEL_FORWARD;
};

#endif // MOTOR_HPP
