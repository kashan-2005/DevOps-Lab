import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { FEATURES, GROUPS, Feature } from './feature-schema';
import { PredictionService, PredictionResponse } from './prediction.service';

const MODEL_NAMES: Record<string, string> = {
  naive_bayes: 'Naive Bayes',
  random_forest: 'Random Forest',
  svc: 'Support Vector',
};

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private api = inject(PredictionService);

  readonly groups = GROUPS.map(g => ({ name: g, fields: FEATURES.filter(f => f.group === g) }));
  readonly fieldCount = FEATURES.length;
  readonly staticDemo = !['localhost', '127.0.0.1'].includes(window.location.hostname);

  values: Record<string, string | number | null> = {};
  readonly loading = signal(false);
  readonly error = signal('');
  readonly result = signal<PredictionResponse | null>(null);

  constructor() {
    this.clear();
  }

  modelName(key: string): string {
    return MODEL_NAMES[key] ?? key;
  }

  voteList(r: PredictionResponse) {
    return Object.entries(r.votes).map(([key, vote]) => ({ key, vote }));
  }

  hint(f: Feature): string {
    const range = f.kind === 'number' ? `${f.min} to ${f.max}` : '';
    return [f.unit, range].filter(Boolean).join(', ');
  }

  fillSample() {
    for (const f of FEATURES) this.values[f.name] = f.sample;
    this.error.set('');
  }

  clear() {
    for (const f of FEATURES) this.values[f.name] = null;
    this.result.set(null);
    this.error.set('');
  }

  submit() {
    this.error.set('');
    this.result.set(null);
    if (this.staticDemo) {
      this.error.set('Predictions run on the local Flask server. Start Flask and Angular on your computer to try it.');
      return;
    }
    const payload: Record<string, string | number> = {};
    for (const f of FEATURES) {
      const value = this.values[f.name];
      if (value === null || value === '') {
        this.error.set(`Enter a value for ${f.label}.`);
        return;
      }
      if (f.kind === 'number') {
        const n = Number(value);
        if (!Number.isFinite(n)) { this.error.set(`${f.label} must be a number.`); return; }
        if (n < (f.min ?? -Infinity) || n > (f.max ?? Infinity)) {
          this.error.set(`${f.label} must be between ${f.min} and ${f.max}.`);
          return;
        }
        payload[f.name] = n;
      } else {
        payload[f.name] = String(value);
      }
    }
    this.loading.set(true);
    this.api.predict(payload).subscribe({
      next: r => { this.result.set(r); this.loading.set(false); },
      error: err => {
        const detail = err.error?.details?.join('; ') || err.error?.error;
        this.error.set(err.status === 0 || err.status >= 500 && !detail
          ? 'Cannot reach the Flask server. Check that python backend/app.py is running.'
          : `The server rejected the request: ${detail || err.message}`);
        this.loading.set(false);
      },
    });
  }
}
