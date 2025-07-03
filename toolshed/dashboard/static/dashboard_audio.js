// dashboard_audio.js - optional audio layer for Toolshed Dashboard
// Requires Tone.js (https://tonejs.github.io/) loaded before this script.
//
// This file intentionally avoids modifying the main dashboard code.
// It monkey-patches ActorDashboard.updateActorStates to receive updates
// and handles start/stop of notes for actors that are in the "working"
// state. Each actor is assigned a note from the C-major scale; if notes
// run out we continue to the next octave.

(() => {
    if (typeof Tone === "undefined") {
        console.warn("Tone.js not found – dashboard audio disabled.");
        return;
    }

    // ---- Configuration ----
    const BASE_OCTAVE = 4;          // Starting octave (C4)
    const NUM_OCTAVES = 3;          // How many octaves to cycle through
    const SCALE = ['C', 'D', 'E', 'F', 'G', 'A', 'B']; // C-major
    const NOTE_DURATION_SEC = 60;   // We will manually release, but give long default
    const DEBUG = true;

    // ---- Generate note pool ----
    const notesPool = [];
    for (let octave = BASE_OCTAVE; octave < BASE_OCTAVE + NUM_OCTAVES; octave++) {
        for (const n of SCALE) {
            notesPool.push(`${n}${octave}`);
        }
    }

    // Tone synth instance (created lazily so the AudioContext isn't touched
    // until the user explicitly enables audio).
    let synth = null;

    // Mapping: actorId -> { note, playing }
    const actorsMap = {};
    let noteIndex = 0;

    let audioEnabled = false;  // user toggle
    let audioUnlocked = false; // browser gesture satisfied

    function stopAllNotes() {
        if (!synth) return;
        for (const meta of Object.values(actorsMap)) {
            if (meta.playing) {
                synth.triggerRelease(meta.note);
                meta.playing = false;
            }
        }
    }

    function updateToggleButton() {
        const btn = document.getElementById('audio-toggle');
        if (!btn) return;
        if (audioEnabled && audioUnlocked) {
            btn.classList.add('enabled');
            btn.textContent = '🔊';
            btn.title = 'Disable sound';
        } else {
            btn.classList.remove('enabled');
            btn.textContent = '🔈';
            btn.title = 'Enable sound';
        }
    }

    function enableAudio() {
        if (!audioUnlocked) {
            Tone.start().then(() => {
                if (!synth) {
                    synth = new Tone.PolySynth(Tone.Synth, {
                        oscillator: { type: 'sawtooth' },
                        volume: 3,
                        envelope: { attack: 0.02, decay: 0.1, sustain: 0.8, release: 0.5 }
                    }).toDestination();
                    window.synth = synth; // allow manual testing in console
                }
                audioUnlocked = true;
                console.log('Audio context started');
                audioEnabled = true;
                updateToggleButton();
                // Play any currently working actors immediately
                if (latestStates && Object.keys(latestStates).length) {
                    AudioManager.handleUpdate(latestStates);
                }
            }).catch(err => {
                console.warn('Failed to start audio context', err);
            });
        } else {
            audioEnabled = true;
            updateToggleButton();
            if (latestStates && Object.keys(latestStates).length) {
                AudioManager.handleUpdate(latestStates);
            }
        }
    }

    function disableAudio() {
        audioEnabled = false;
        stopAllNotes();
        updateToggleButton();
    }

    document.addEventListener('DOMContentLoaded', () => {
        const btn = document.getElementById('audio-toggle');
        if (btn) {
            btn.addEventListener('click', () => {
                if (audioEnabled) {
                    disableAudio();
                } else {
                    enableAudio();
                }
            });
        }
        updateToggleButton();
    });

    // ---- Audio Manager ----
    let latestStates = {};

    const AudioManager = {
        handleUpdate(flatStates) {
            latestStates = flatStates;
            // flatStates is an object: { actorId: { id, state, ... } }
            const activeActors = new Set();

            for (const [actorId, actor] of Object.entries(flatStates)) {
                activeActors.add(actorId);
                const isWorking = actor.state === 'working';

                // Ensure actor has a note
                if (!actorsMap[actorId]) {
                    actorsMap[actorId] = {
                        note: notesPool[noteIndex % notesPool.length],
                        playing: false,
                    };
                    noteIndex++;
                }
                const meta = actorsMap[actorId];

                if (audioEnabled && audioUnlocked && isWorking && !meta.playing) {
                    if (!synth) return; // safety
                    synth.triggerAttack(meta.note, undefined, 0.8);
                    if (DEBUG) console.log(`▶️  attack ${meta.note} (actor ${actorId})`);
                    meta.playing = true;
                } else if ((!isWorking || !audioEnabled || !audioUnlocked) && meta.playing) {
                    if (!synth) return;
                    synth.triggerRelease(meta.note);
                    if (DEBUG) console.log(`⏹️  release ${meta.note} (actor ${actorId}) reason=${isWorking? 'audioDisabled':'notWorking'}`);
                    meta.playing = false;
                }
            }

            // Stop notes for actors that disappeared or audio disabled
            for (const [actorId, meta] of Object.entries(actorsMap)) {
                if ((!activeActors.has(actorId) || !audioEnabled || !audioUnlocked) && meta.playing) {
                    if (!synth) return;
                    synth.triggerRelease(meta.note);
                    if (DEBUG) console.log(`⏹️  release ${meta.note} (actor ${actorId}) reason=actorGoneOrMuted`);
                    meta.playing = false;
                }
            }
        }
    };

    // ---- Monkey-patch ActorDashboard ----
    const patchDashboard = () => {
        if (!window.dashboard) {
            return false; // Need dashboard instance
        }

        // Determine target object to patch
        const target = window.ActorDashboard ? window.ActorDashboard.prototype : window.dashboard;
        if (target.__audioPatched) return true;

        if (DEBUG) console.log('🔧 Patching', window.ActorDashboard ? 'ActorDashboard prototype' : 'dashboard instance', 'for audio');

        const originalUpdate = target.updateActorStates;
        if (typeof originalUpdate !== 'function') {
            console.warn('updateActorStates not found on target');
            return false;
        }

        target.updateActorStates = function(states) {
            // Call original (ensure correct this binding)
            originalUpdate.call(this, states);

            // Flatten actor states (copied from dashboard.js)
            const flatActors = {};
            for (const actors of Object.values(states)) {
                for (const actor of actors) {
                    flatActors[actor.id] = actor;
                }
            }
            // Delegate to audio manager
            if (DEBUG) console.log('🎛 handleUpdate called with', Object.keys(flatActors).length, 'actors');
            AudioManager.handleUpdate(flatActors);
        };
        if (DEBUG) console.log('✅ Audio patch applied');
        target.__audioPatched = true;
        return true;
    };

    // ---- Wait for user interaction to unlock audio ----
    // Remove global click unlock listener and interval remains as patchDashboard
    // Adapt patchDashboard and rest unchanged.

    // ---- Retry patching until dashboard is ready ----
    const interval = setInterval(() => {
        if (DEBUG) {
            if (typeof window !== 'undefined') {
                console.log('[audio] patch attempt - ActorDashboard:', !!window.ActorDashboard, 'dashboard:', !!window.dashboard);
            }
        }
        if (patchDashboard()) clearInterval(interval);
    }, 100);

    console.log('Dashboard audio layer loaded');
})(); 