export interface FoodSearchResult {
  fdcId: number;
  description: string;
  dataType: string;
  gtinUpc?: string;
  publishedDate: string;
  brandOwner?: string;
  ingredients?: string;
  servingSize?: number;
  servingSizeUnit?: string;
  foodNutrients: FoodNutrient[];
}

export interface FoodNutrient {
  nutrientId: number;
  nutrientName: string;
  nutrientNumber: string;
  unitName: string;
  value: number;
}

export interface SearchResponse {
  foods: FoodSearchResult[];
  totalHits: number;
  currentPage: number;
  totalPages: number;
  pageSize: number;
}

export interface FoodDetailResponse {
  fdcId: number;
  description: string;
  dataType: string;
  gtinUpc?: string;
  publishedDate: string;
  brandOwner?: string;
  ingredients?: string;
  servingSize?: number;
  servingSizeUnit?: string;
  foodNutrients: FoodNutrient[];
  foodComponents?: FoodComponent[];
  foodAttributes?: FoodAttribute[];
  foodPortions?: FoodPortion[];
  inputFoods?: InputFood[];
  measures?: Measure[];
  labelNutrients?: LabelNutrient[];
}

export interface FoodComponent {
  id: number;
  name: string;
  dataPoints?: number;
  gramWeight?: number;
  dataType?: string;
  isRefuse?: boolean;
  amount?: number;
  unitName?: string;
}

export interface FoodAttribute {
  id: number;
  sequenceNumber?: number;
  value: string;
  foodAttributeType: {
    id: number;
    name: string;
    description?: string;
  };
}

export interface FoodPortion {
  id: number;
  amount?: number;
  dataPoints?: number;
  gramWeight?: number;
  modifier?: string;
  measureUnit: {
    id: number;
    name: string;
    abbreviation?: string;
  };
  sequenceNumber?: number;
  minYearAcquired?: number;
}

export interface InputFood {
  id: number;
  amount?: number;
  unit: {
    id: number;
    name: string;
    abbreviation?: string;
  };
  foodDescription?: string;
  fdcId?: number;
  dataType?: string;
}

export interface Measure {
  id: number;
  amount?: number;
  measureUnit: {
    id: number;
    name: string;
    abbreviation?: string;
  };
  qualifiers?: {
    id: number;
    name: string;
  }[];
}

export interface LabelNutrient {
  id: number;
  amount: number;
  nutrient: {
    id: number;
    name: string;
    unitName: string;
  };
}
