// Listen for messages from content script
// chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
//   if (request.greeting === "hello") {
//     // Query all tabs and send a message to all content scripts
//     chrome.tabs.query({}, function (tabs) {
//       for (let tab of tabs) {
//         chrome.tabs.sendMessage(tab.id, { greeting: "hello" }, function (response) {
//           // Handle the response if needed
//           console.log(response ? response.farewell : "No response from tab " + tab.id);
//         });
//       }
//     });
//     sendResponse({ farewell: "Message broadcasted to all tabs" });
//   }
// });

chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  console.log("starting ... on message")
  if (message.currentUrl) {
    console.log("Current URL:", message.currentUrl);
    // Perform actions with the received URL
    
    // chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    //   chrome.tabs.sendMessage(tabs[0].id, { action: "updateUrl", url: message.currentUrl });
    // });
  }

  if (message.type === "showNotification") {
    console.log("starting notification ... ")
    chrome.notifications.create({
        type: "basic",
        iconUrl: "images/icon48.png", // Icon URL for the notification
        title: "Message Received",
        message: message.text
    });
  }

  if (request.action === 'getVar') {
    console.log("Should have saved.")
    sendResponse({ data: myVar });
  }
  
});

