#include <ESP8266WiFi.h>
#include <PubSubClient.h>

#define REDLED D3
#define GREENLED D1

// WiFi
const char *ssid = "Subhajit"; // Enter your WiFi name
const char *password = "buzz_Lightyear@10";  // Enter WiFi password

// MQTT Broker
const char *mqtt_broker = "192.168.1.7";
const char *topic = "esp8266/test";
const char *publishTopic = "esp8266/device";
const int mqtt_port = 1883;

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  // Set software serial baud to 115200;
  pinMode(REDLED, OUTPUT);
  pinMode(GREENLED, OUTPUT);
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
      delay(500);
  }
  
  client.setServer(mqtt_broker, mqtt_port);
  client.setCallback(callback);
  while (!client.connected()) {
      String client_id = "esp8266-client-";
      client_id += String(WiFi.macAddress());
      if (!client.connect(client_id.c_str())) {
         delay(2000);
      } 
  }
  // publish and subscribe
  client.subscribe(topic);
  client.publish(publishTopic, String("Ready").c_str(), true);
}

void callback(char *topic, byte *payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
      message += (char)payload[i];
  }
  String messageString = message;
  if(message == "Mask") {
    Serial.println("Mask");
    digitalWrite(REDLED, LOW);
    digitalWrite(GREENLED, HIGH);
    delay(2000);
    digitalWrite(GREENLED, LOW);
    client.publish(publishTopic, String("Ready").c_str(), true);
  }
  
}

void loop() {
  digitalWrite(REDLED, HIGH);
  client.loop();
}
