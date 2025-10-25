from src.device_manager import DeviceManager
from src.ui import ChatbotUI

if __name__ == "__main__":
    device = DeviceManager().device
    app = ChatbotUI(device=device)
    app.launch()
