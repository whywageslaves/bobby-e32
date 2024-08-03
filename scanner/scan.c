/* Scan Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

/*
    This example shows how to scan for available set of APs.
*/

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "esp_event.h"
#include "nvs_flash.h"

#define DEFAULT_SCAN_LIST_SIZE CONFIG_EXAMPLE_SCAN_LIST_SIZE

#define BEACON_NUM 7
#define BEACON_SSID_MAX_LEN 32
// A magic number to indicate if no rssi value was measured for a beacon.
#define OUT_OF_RANGE 42

static const char *TAG = "scan";

typedef struct BeaconInfo {
    uint8_t channel;
    uint8_t ssid[BEACON_SSID_MAX_LEN];
} BeaconInfo;

// Initialize Wi-Fi as sta and set scan method
static void wifi_scan(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();
    assert(sta_netif);

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Find currently configured maximum transmit power.
    int8_t tx_power;
    ESP_ERROR_CHECK(esp_wifi_get_max_tx_power(&tx_power));
    ESP_LOGI(TAG, "Max tx power: [%d]\n", tx_power);

    // Change power mode to non_power saving - dunno if this changes anything.
    const wifi_ps_type_t ps_mode = WIFI_PS_NONE;
    ESP_ERROR_CHECK(esp_wifi_set_ps(ps_mode));

    BeaconInfo beacons[BEACON_NUM] = {
        {
            .ssid = {'b', 'o', 'b', '1', '\0'},
            .channel = 1
        },
        {
            .ssid = {'b', 'o', 'b', '2', '\0'},
            .channel = 6
        },
        {
            .ssid = {'b', 'o', 'b', '3', '\0'},
            .channel = 11
        },
        {
            .ssid = {'b', 'o', 'b', '4', '\0'},
            .channel = 4
        },
        {
            .ssid = {'b', 'o', 'b', '5', '\0'},
            .channel = 5
        },
        {
            .ssid = {'b', 'o', 'b', '6', '\0'},
            .channel = 6
        },
        {
            .ssid = {'b', 'o', 'b', '7', '\0'},
            .channel = 7
        },
    };

    // uint8_t BEACON_SSIDS[BEACON_NUM][BEACON_SSID_MAX_LEN] = { 
    //     {'b', 'o', 'b', '1', '\0'},
    //     {'b', 'o', 'b', '2', '\0'},
    //     {'b', 'o', 'b', '3', '\0'},
    // };

    // Scan time per channel, in millis.
    const wifi_scan_time_t scan_time = {
        .active = {
            .min = 0,
            .max = 120,
        },
        .passive = 120,
    };

    wifi_scan_config_t scan_conf;
    scan_conf.bssid = NULL;
    scan_conf.show_hidden = false;
    scan_conf.scan_type = WIFI_SCAN_TYPE_ACTIVE;
    // scan_conf.scan_type = WIFI_SCAN_TYPE_PASSIVE;
    scan_conf.scan_time = scan_time;

    ESP_LOGI(TAG, "Scanning Start.\n");

    while (1) {
        for (int i = 0; i < BEACON_NUM; ++i) {
            // Change the scan config to selectively scan
            // for a specific ssid on a specific channel.
            scan_conf.ssid = beacons[i].ssid;
            scan_conf.channel = beacons[i].channel;
            // ESP_LOGI("info", "i: %d; channel: %d; ssid: %s", i, scan_conf.channel, scan_conf.ssid);

            // Holds the results of our wifi Access Point (AP) scan.
            wifi_ap_record_t ap_info;
            esp_err_t scan_res = esp_wifi_scan_start(&scan_conf, true);
            if (scan_res != ESP_OK) {
                ESP_LOGI(TAG, "scan_start failed.");
            }

            // uint16_t record_num = 1;
            // NOTE: Don't use ap_records. It never fails, and always
            // returns the last valid scan result? (documentation is incorrect).
            esp_err_t res = esp_wifi_scan_get_ap_record(&ap_info);
            if (res != ESP_OK) {
                // Treat all errors as failure.
                // NOTE: ESP_LOGI seems to add a newline character to the msg, 
                // so don't add a newline yourself.

                // Obviously, since we haven't found an entry, can't use 
                // ap_info.ssid. Use BEACON_SSIDS[i] instead.
                // ESP_LOGI(TAG, "lolwut");
                ESP_LOGI(TAG, "ssid:%s;rssi:%d", beacons[i].ssid, OUT_OF_RANGE);
                // ESP_LOGI(TAG, "ssid:%s;rssi:%d", beacons[i].ssid, OUT_OF_RANGE);
            } else {
                ESP_LOGI(TAG, "ssid:%s;rssi:%d", ap_info.ssid, ap_info.rssi);
            }
        }
    }
}

void app_main(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK( ret );
    wifi_scan();
}


// /* Scan Example

//    This example code is in the Public Domain (or CC0 licensed, at your option.)

//    Unless required by applicable law or agreed to in writing, this
//    software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//    CONDITIONS OF ANY KIND, either express or implied.
// */

// /*
//     This example shows how to scan for available set of APs.
// */
// #include <string.h>
// #include "freertos/FreeRTOS.h"
// #include "freertos/event_groups.h"
// #include "esp_wifi.h"
// #include "esp_log.h"
// #include "esp_event.h"
// #include "nvs_flash.h"

// #define DEFAULT_SCAN_LIST_SIZE 20

// static const char *TAG = "scan";

// /* Initialize Wi-Fi as sta and set scan method */
// static void wifi_scan(void)
// {
//     ESP_ERROR_CHECK(esp_netif_init());
//     ESP_ERROR_CHECK(esp_event_loop_create_default());
//     esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();
//     assert(sta_netif);

//     wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
//     ESP_ERROR_CHECK(esp_wifi_init(&cfg));

//     uint16_t number = DEFAULT_SCAN_LIST_SIZE;
//     wifi_ap_record_t ap_info[DEFAULT_SCAN_LIST_SIZE];

//     ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
//     ESP_ERROR_CHECK(esp_wifi_start());
//     while(1) {
//         esp_wifi_scan_start(NULL, true);
//         uint16_t ap_count = 0;
//         memset(ap_info, 0, sizeof(ap_info));

//         uint16_t scan_num = DEFAULT_SCAN_LIST_SIZE;
//         ESP_ERROR_CHECK(esp_wifi_scan_get_ap_records(&scan_num, ap_info));
//         // ESP_LOGI(TAG, "number: %d\n", scan_num);
//         // ESP_ERROR_CHECK(esp_wifi_scan_get_ap_num(&ap_count));
//         ESP_LOGI(TAG, "Total APs scanned = %u", scan_num);
//         for (int i = 0; (i < DEFAULT_SCAN_LIST_SIZE) && (i < scan_num); i++) {
//             ESP_LOGI(TAG, "SSID \t\t%s", ap_info[i].ssid);
//             ESP_LOGI(TAG, "RSSI \t\t%d", ap_info[i].rssi);
//             ESP_LOGI(TAG, "Channel \t\t%d\n", ap_info[i].primary);
//         }
//         // vTaskDelay(1000 / portTICK_PERIOD_MS);
//     }
// }

// void app_main(void)
// {
//     // Initialize NVS
//     esp_err_t ret = nvs_flash_init();
//     if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
//         ESP_ERROR_CHECK(nvs_flash_erase());
//         ret = nvs_flash_init();
//     }
//     ESP_ERROR_CHECK( ret );
//     wifi_scan();
// }
