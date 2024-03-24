<template>
    <form @submit.prevent="sendFile()" class="form-wrapper d-flex flex-column justify-content-start align-items-center me-auto p-4" ref="form">
        <h2 class="tl">Upload</h2>
        <div @click="triggerFileinput" class="upload-trigger d-flex justify-content-center align-items-center position-relative">
            <p class="m-0 pe-none">Upload .csv</p>
        </div>
        <input type="file" id="fileupload" name="fileupload" class="d-none" ref="fileInput">
        <input type="submit" value="Submit">
    </form>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const fileInput = ref<HTMLInputElement>();

const form = ref<HTMLFormElement>();
function sendFile() {
    console.clear();
    console.dir(form.value);
    console.dir(form.value!.files);
    fetch("http://localhost:8000")
    .then(response => {
        console.log(response);
    });
}

function triggerFileinput(e: MouseEvent) {
    fileInput.value!.click();
}
</script>

<style scoped>
.upload-trigger:hover {
    color: #c7c7c7;
}
.upload-trigger:hover::after,
.upload-trigger:hover::before {
    height: 50%;
}
.upload-trigger::after {
    bottom: 0;
}
.upload-trigger::before {
    top: 0;
}
.upload-trigger::before,
.upload-trigger::after {
    content: "";
    position: absolute;
    width: 100%;
    height: 0;
    z-index: -1;
    left: 0;
    background-color: var(--dark);
    transition: all .3s ease;
}
.upload-trigger {
    margin: 50px 0;
    width: 100px;
    height: 40px;
    border: 3px solid black;
    cursor: pointer;
    transition: all .3s ease;
}
.form-wrapper {
    min-width: 450px;
    margin-top: 150px;
    border-radius: 15px;
    background-color: rgba(255,255,255,.1);
    box-shadow: 0 0 10px 2px rgba(255,255,255,.2);
}
</style>