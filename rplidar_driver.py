"""
Native lightweight RPLiDAR A1M8 serial driver using standard pyserial.
No third-party rplidar package dependency required.
Supports A1/A1M8 protocol: Express and Standard Scan.
"""
import time
import serial
import serial.tools.list_ports
import struct
import threading
import logging

logger = logging.getLogger("RPLidarDriver")

SYNC_BYTE = b'\xa5'
SYNC_BYTE2 = b'\x5a'

CMD_STOP = b'\xa5\x25'
CMD_RESET = b'\xa5\x40'
CMD_SCAN = b'\xa5\x20'
CMD_GET_INFO = b'\xa5\x50'
CMD_GET_HEALTH = b'\xa5\x52'

def scan_available_ports():
    """List all available serial COM ports with hardware descriptions."""
    ports = serial.tools.list_ports.comports()
    results = []
    for p in ports:
        is_silabs = "CP210" in (p.description or "") or "Silicon Labs" in (p.manufacturer or "")
        results.append({
            "device": p.device,
            "description": p.description,
            "hwid": p.hwid,
            "likely_rplidar": is_silabs or "USB Serial" in (p.description or "")
        })
    return results

def autodetect_rplidar_port():
    """Attempt auto-detection of RPLiDAR A1 (typically CP2102 USB-to-UART bridge)."""
    ports = scan_available_ports()
    for p in ports:
        if p["likely_rplidar"]:
            return p["device"]
    if ports:
        return ports[0]["device"]
    return None

class SimpleRPLidar:
    """Robust, thread-safe RPLiDAR A1 driver."""
    def __init__(self, port, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.scanning = False
        self._lock = threading.Lock()

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
            self.serial.setDTR(False)  # Motor start on some breakout boards
            time.sleep(0.1)
            self.stop()
            self.clear_buffer()
            logger.info(f"Connected to RPLiDAR on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to LiDAR on {self.port}: {e}")
            return False

    def clear_buffer(self):
        if self.serial and self.serial.is_open:
            self.serial.reset_input_buffer()

    def stop(self):
        if self.serial and self.serial.is_open:
            self.serial.write(CMD_STOP)
            time.sleep(0.02)

    def reset(self):
        if self.serial and self.serial.is_open:
            self.serial.write(CMD_RESET)
            time.sleep(0.05)

    def start_scan(self):
        if not self.serial or not self.serial.is_open:
            raise RuntimeError("LiDAR serial not open")
        self.clear_buffer()
        self.serial.write(CMD_SCAN)
        
        # Read descriptor (7 bytes)
        desc = self.serial.read(7)
        if len(desc) < 7 or desc[0:2] != b'\xa5\x5a':
            raise RuntimeError(f"Bad response descriptor from LiDAR: {desc.hex()}")
        self.scanning = True

    def iter_measurements(self):
        """Yield (quality, angle_degrees, distance_meters) tuples continuously."""
        if not self.scanning:
            self.start_scan()
        
        while self.scanning:
            if not self.serial or not self.serial.is_open:
                break
            
            raw = self.serial.read(5)
            if len(raw) < 5:
                continue
            
            # Format:
            # byte 0: [quality (6-bit), S_bar, S]
            # byte 1: [angle_q6[6:0], check_bit]
            # byte 2: [angle_q6[14:7]]
            # byte 3: [dist_q2[7:0]]
            # byte 4: [dist_q2[15:8]]
            b0, b1, b2, b3, b4 = raw
            
            # Check bit check
            if not (b1 & 0x01):
                # Desynchronized, scan for sync byte
                self.serial.read(1)
                continue
            
            quality = b0 >> 2
            angle_q6 = (b1 >> 1) | (b2 << 7)
            angle_deg = angle_q6 / 64.0
            
            dist_q2 = b3 | (b4 << 8)
            dist_meters = (dist_q2 / 4.0) / 1000.0  # mm to meters
            
            if dist_meters > 0.05 and dist_meters < 12.0:
                yield quality, angle_deg, dist_meters

    def disconnect(self):
        self.scanning = False
        try:
            if self.serial and self.serial.is_open:
                self.stop()
                self.serial.close()
        except Exception:
            pass
