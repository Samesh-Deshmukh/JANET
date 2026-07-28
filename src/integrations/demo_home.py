# src/integrations/demo_home.py
"""A fake smart home with a handful of believable devices, for demos and
end-to-end testing. Selected with JANET_SMART_HOME=demo. Nothing real switches on."""
from integrations.smart_home_source import Device


def demo_devices():
    """Sample devices covering the three domains JANET controls, with a mix of
    starting states so a demo can both turn something on and turn something off."""
    return [
        Device("Living Room Lights", "light.living_room", "off", "light"),
        Device("Bedroom Lamp", "light.bedroom_lamp", "off", "light"),
        Device("Kitchen Lights", "light.kitchen", "on", "light"),
        Device("Desk Lamp", "light.desk_lamp", "off", "light"),
        Device("Fan", "fan.bedroom_fan", "off", "fan"),
        Device("Coffee Maker", "switch.coffee_maker", "off", "switch"),
    ]
