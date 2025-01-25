from midiutil.MidiFile import MIDIFile
import random

# Extended chord types
MAJOR_TRIAD = [0, 4, 7]          # Major
MINOR_TRIAD = [0, 3, 7]          # Minor
DIM_TRIAD = [0, 3, 6]            # Diminished
AUG_TRIAD = [0, 4, 8]            # Augmented
MAJ7 = [0, 4, 7, 11]             # Major 7th
MIN7 = [0, 3, 7, 10]             # Minor 7th
DOM7 = [0, 4, 7, 10]             # Dominant 7th
DIM7 = [0, 3, 6, 9]              # Diminished 7th

# Keys (starting from C4 = 60)
ROOT_NOTES = {
    'C': 60, 'C#': 61, 'Db': 61,
    'D': 62, 'D#': 63, 'Eb': 63,
    'E': 64,
    'F': 65, 'F#': 66, 'Gb': 66,
    'G': 67, 'G#': 68, 'Ab': 68,
    'A': 69, 'A#': 70, 'Bb': 70,
    'B': 71
}

def create_chord(root_note, chord_type, inversion=0):
    chord = [root_note + interval for interval in chord_type]
    # Apply inversion
    for _ in range(inversion):
        chord = chord[1:] + [chord[0] + 12]
    return chord

def create_progression(key='C', progression_type='basic'):
    progressions = {
        'basic': [(0, MAJOR_TRIAD), (5, MAJOR_TRIAD), (3, MINOR_TRIAD), (4, MAJOR_TRIAD)],  # I-V-iii-IV
        'jazz': [(0, MAJ7), (3, MIN7), (6, DOM7), (2, MIN7)],  # Imaj7-IVm7-VII7-IIIm7
        'pop': [(0, MAJOR_TRIAD), (5, MAJOR_TRIAD), (3, MINOR_TRIAD), (4, MAJOR_TRIAD)]  # I-V-iii-IV
    }
    
    root = ROOT_NOTES[key]
    chosen_prog = progressions[progression_type]
    return [(root + interval, chord_type) for interval, chord_type in chosen_prog]

def generate_practice_sequence(key='C', progression_type='basic', duration=480):
    sequence = []
    progression = create_progression(key, progression_type)
    
    for root, chord_type in progression:
        # Create basic chord
        chord = create_chord(root, chord_type)
        # Add as block chord
        sequence.extend([(note, duration) for note in chord])
        # Add as arpeggio
        sequence.extend([(note, duration//2) for note in chord])
        sequence.extend([(note, duration//2) for note in reversed(chord)])
    
    return sequence

# Create MIDI file
midi = MIDIFile(1)
track = 0
time = 0
tempo = 120

midi.addTempo(track, time, tempo)

# Generate practice sequence
sequence = generate_practice_sequence(key='C', progression_type='basic')

# Add notes to MIDI file
current_time = 0
for note, duration in sequence:
    midi.addNote(track, 0, note, current_time, duration/480, 100)
    current_time += duration/480

# Save MIDI file
with open("/home/dass/Documents/Piano/piano_chord_practice.mid", "wb") as output_file:
    midi.writeFile(output_file)