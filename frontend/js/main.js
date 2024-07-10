// Main js is binded to the main js 
// This function sends a message to the content script to retrieve the URL
function getCurrentTabUrl() {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const currentTab = tabs[0];
    chrome.tabs.sendMessage(currentTab.id, { action: "getCurrentTabUrl" });
  });
}

// This function handles the message received from content script and updates the popup
function updateUrl(url) {
  const urlDisplay = document.getElementById("urlDisplay");
  urlDisplay.textContent = url;
}



document.addEventListener('DOMContentLoaded', function() {
  // Add event listener to the button
  document.getElementById('saveDataBtn').addEventListener('click', function() {
    
      // chrome.storage.local.get(['affected'], function(result) {
      //   console.log('Value currently is ' + result.key);
      //   chrome.storage.local.set({affected : result.key +1}, function() {
      //       alert('Data saved successfully!');
      //   });
      // });
      chrome.storage.local.clear(function() {
        var error = chrome.runtime.lastError;
        if (error) {
            console.error(error);
        }
      });
      
  });
  
});



// On popup load, get the current tab's URL
document.addEventListener("DOMContentLoaded", function () {
  getCurrentTabUrl();
});