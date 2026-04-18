use crate::ToastNotification;

pub fn show_notif(notif: ToastNotification) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("[toast] {}: {}", notif.title, notif.message);
    Ok(())
}
