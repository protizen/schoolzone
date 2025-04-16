#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <ArduinoJson.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_task_wdt.h"

// WiFi 설정
const char* ssid = "class606_2.4G";
const char* password = "sejong123";

// 서버 설정
const char* serverUrl = "http://10.0.66.99:5000/detect";

// LED 핀 설정
#define LED_PIN 4
WebServer server(80);

// 카메라 모델 설정
#define CAMERA_MODEL_AI_THINKER

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define WDT_TIMEOUT_SECONDS 10
TaskHandle_t camTaskHandle = NULL;
SemaphoreHandle_t camSemaphore = NULL;

unsigned long lastResetTime = 0;
int resetCount = 0;
unsigned long wifiReconnectAttempts = 0;

void connectToWiFi() {
  Serial.println("WiFi 연결 시도 중...");
  WiFi.disconnect();
  delay(1000);
  WiFi.begin(ssid, password);
  
  unsigned long startAttemptTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 30000) {
    delay(500);
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi 연결 완료");
    Serial.println(WiFi.localIP());
    wifiReconnectAttempts = 0;
  } else {
    Serial.println("\nWiFi 연결 실패. 재시작합니다.");
    wifiReconnectAttempts++;
    if (wifiReconnectAttempts > 5) {
      Serial.println("WiFi 연결 실패 횟수 과다. 시스템 재시작...");
      ESP.restart();
    }
  }
}

void checkWiFiConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi 연결이 끊어졌습니다. 재연결 시도...");
    connectToWiFi();
  }
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("카메라 초기화 실패, 오류: 0x%x\n", err);
    return false;
  }
  
  Serial.println("카메라 초기화 성공");
  return true;
}

bool dangerAlertActive = false;

void alertDangerTask(void *pvParameters) {
  dangerAlertActive = true;
  for (int i = 0; i < 5; i++) {
    digitalWrite(LED_PIN, HIGH);
    vTaskDelay(100 / portTICK_PERIOD_MS);
    digitalWrite(LED_PIN, LOW);
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
  dangerAlertActive = false;
  vTaskDelete(NULL);
}

void alertDanger() {
  if (!dangerAlertActive) {
    xTaskCreate(alertDangerTask, "AlertDangerTask", 2048, NULL, 1, NULL);
  }
}

HTTPClient http; // HTTPClient 객체를 전역으로 선언

void handleRequest() {
  if (server.method() == HTTP_GET) {
    if (server.hasArg("danger")) {
      String dangerValue = server.arg("danger");
      String personCnt = server.arg("person");
      String vehicleCnt = server.arg("vehicle");
      if (dangerValue == "1") {
        alertDanger();
        Serial.println("서버 응답: [person: " + personCnt + ", vehicle: " + vehicleCnt + "]");
        server.send(200, "text/plain; charset=UTF-8", "위험 상황 알림: LED Light가 깜빡입니다. [person: " + personCnt + ", vehicle: " + vehicleCnt + "]");
      } else {
        server.send(200, "text/plain; charset=UTF-8", "위험 상황 없음.");
        digitalWrite(LED_PIN, LOW);
      }
    } else {
      server.send(400, "text/plain; charset=UTF-8", "잘못된 요청.");
      digitalWrite(LED_PIN, LOW);
    }
  } else {
    server.send(200, "text/plain; charset=UTF-8", "GET 요청 처리됨.");
  }
}

void cameraTask(void *parameter) {
  esp_task_wdt_add(NULL);

  while (true) {
    esp_task_wdt_reset();

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi 연결 끊김. 재연결 시도...");
      connectToWiFi();
      vTaskDelay(1000 / portTICK_PERIOD_MS);
      continue;
    }

    camera_fb_t *fb = NULL;

    if (xSemaphoreTake(camSemaphore, portMAX_DELAY) == pdTRUE) {
      fb = esp_camera_fb_get();
      xSemaphoreGive(camSemaphore);
    }

    if (!fb) {
      Serial.println("사진 촬영 실패");
      vTaskDelay(1000 / portTICK_PERIOD_MS);
      continue;
    }

    String url = String(serverUrl) + "?ip=" + WiFi.localIP().toString();
    http.begin(url); // URL 설정
    http.addHeader("Content-Type", "image/jpeg");

    int httpResponseCode = http.POST(fb->buf, fb->len);

    if (httpResponseCode > 0) {
      String response = http.getString();
      DynamicJsonDocument doc(1024);
      DeserializationError error = deserializeJson(doc, response);

      if (!error) {
        bool isDanger = doc["danger"];
        if (isDanger) {
          alertDanger();
        } else {
          digitalWrite(LED_PIN, LOW);
        }
      } else {
        Serial.println("JSON 파싱 오류");
      }
    } else {
      Serial.print("HTTP 에러 코드: ");
      Serial.println(httpResponseCode);
    }

    http.end(); // HTTP 요청 종료
    esp_camera_fb_return(fb);
    Serial.print("Free heap: ");
    Serial.println(ESP.getFreeHeap());
    vTaskDelay(500 / portTICK_PERIOD_MS);
  }
}

void webServerTask(void *parameter) {
  esp_task_wdt_add(NULL);
  while (true) {
    esp_task_wdt_reset();
    server.handleClient();
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

void setupWatchdog() {
  Serial.println("워치독 타이머 설정 중...");
  esp_task_wdt_config_t wdtConfig;
  wdtConfig.timeout_ms = WDT_TIMEOUT_SECONDS * 1000;
  wdtConfig.idle_core_mask = (1 << 0) | (1 << 1);
  wdtConfig.trigger_panic = true;

  esp_err_t err = esp_task_wdt_init(&wdtConfig);
  if (err != ESP_OK) {
    Serial.printf("워치독 타이머 초기화 실패: %d\n", err);
    return;
  }

  Serial.println("워치독 타이머 초기화 성공");
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  
  Serial.begin(115200);
  Serial.println("시스템 시작");
  delay(1000);
  
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  setupWatchdog();
  
  camSemaphore = xSemaphoreCreateMutex();
  
  if (!initCamera()) {
    for (int i = 0; i < 10; i++) {
      digitalWrite(LED_PIN, HIGH);
      delay(100);
      digitalWrite(LED_PIN, LOW);
      delay(100);
    }
    ESP.restart();
  }
  
  connectToWiFi();
  
  server.on("/", HTTP_GET, handleRequest);
  server.on("/", HTTP_POST, handleRequest);
  server.begin();
  
  xTaskCreatePinnedToCore(
    cameraTask,
    "CameraTask",
    8192,
    NULL,
    2,
    &camTaskHandle,
    0
  );
  
  xTaskCreatePinnedToCore(
    webServerTask,
    "WebServerTask",
    4096,
    NULL,
    1,
    NULL,
    1
  );
  
  esp_task_wdt_add(NULL);
  
  lastResetTime = millis();
}

void loop() {
  esp_task_wdt_reset();
  
  checkWiFiConnection();
  
  static unsigned long lastMemCheck = 0;
  if (millis() - lastMemCheck > 20000) {
    lastMemCheck = millis();
    Serial.printf("시스템 정보 - 업타임: %lu초, 메모리: %lu bytes\n", 
                 millis() / 1000, ESP.getFreeHeap());
  }
  
  if (millis() - lastResetTime > 7200000) {
    Serial.println("정기 유지보수 재시작 수행");
    ESP.restart();
  }
  
  delay(1000);
}