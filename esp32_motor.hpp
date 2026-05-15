#ifndef ESP32_MOTOR_HPP
#define ESP32_MOTOR_HPP

#include <string>
#include <stdint.h>

class Esp32Motor {
public:
    Esp32Motor(const char* uart_dev = "/dev/ttyS1", int baud = 115200);
    ~Esp32Motor();

    bool init();
    void forward(int speed);
    void backward(int speed);
    void left(int speed);
    void right(int speed);
    void brake();
    void standby();
    // 差速驱动：正值前进，负值后退，范围 [-255, 255]
    void drive(int left_speed, int right_speed);

private:
    bool send_frame(uint8_t cmd, const uint8_t* payload = nullptr, uint8_t len = 0);
    bool recv_ack(uint8_t expected_cmd, int timeout_ms = 200);
    uint8_t calc_chk(uint8_t cmd, uint8_t len, const uint8_t* payload);

    int uart_fd_;
    std::string uart_dev_;
    int baud_;
    bool initialized_;
};

#endif // ESP32_MOTOR_HPP