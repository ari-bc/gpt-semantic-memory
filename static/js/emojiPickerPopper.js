const button = document.querySelector('#emoji-button');
const pickerContainer = document.querySelector('#emoji-picker-container');
const picker = pickerContainer.querySelector('emoji-picker');

picker.addEventListener('emoji-click', (event) => {
  document.querySelector('#user-input').value += event.detail.unicode;
});

const popperInstance = Popper.createPopper(button, pickerContainer, {
  modifiers: [
    {
      name: 'offset',
      options: {
        offset: [0, 10],
      },
    },
  ],
});

window.toggleEmojiPicker = function() {
  pickerContainer.classList.toggle('emoji-picker-hidden');
}

document.addEventListener('click', (event) => {
  if (event.target !== button && !pickerContainer.contains(event.target)) {
    pickerContainer.classList.add('emoji-picker-hidden');
  }
});
