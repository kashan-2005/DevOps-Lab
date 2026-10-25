import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResponse {
  prediction: 'healthy' | 'diseased';
  votes: Record<string, 'healthy' | 'diseased'>;
  agreement: string;
  task: string;
  disclaimer?: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private http = inject(HttpClient);

  // Relative URL: the Angular dev server proxies /api -> Flask on 127.0.0.1:5000
  predict(payload: Record<string, string | number>): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>('/api/predict', payload);
  }
}
