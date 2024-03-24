<template>
  <div class="container d-flex flex-column justify-content-start align-items-center pt-5 border border-danger">
    <h1 class="app-headline me-auto">TIMESERIES PREDICTOR</h1>
    <the-upload></the-upload>
  </div>
</template>

<script setup lang="ts">
import TheUpload from './components/TheUpload.vue';
</script>

<style>
.app-headline {
  font-size: 60px;
}
</style>
