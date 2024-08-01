// This function handles the message received from content script and updates the popup
function updateUrl(url) {
  const urlDisplay = document.getElementById("urlDisplay");
  urlDisplay.textContent = url;
}



document.addEventListener('DOMContentLoaded', function() {

  document.getElementById('resetDataBtn').addEventListener('click', function() {
    
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
  
  document.getElementById('blockSiteBtn').addEventListener('click', async function() {

    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      // Send a message to the content script in the active tab
      chrome.tabs.sendMessage(tabs[0].id, { message: "urlBlock" }, function (response) {
        console.log(response.farewell);
      });
    });
  });
    
});
