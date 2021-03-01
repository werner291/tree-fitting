use std::f32::INFINITY;

use image::{Rgba, RgbaImage};
use nalgebra::Vector2;
use ndarray::Array2;
use ndarray_stats::QuantileExt;

fn central_gradient_x(field: &Array2<f32>) -> Array2<f32> {
    let mut res: Array2<f32> = Array2::zeros(field.dim());

    // bulk
    {
        let mut s = res.slice_mut(s![1..-1, ..]);
        s += &field.slice(s![2.., ..]);
        s -= &field.slice(s![..-2, ..]);
    }
    // borders
    {
        let mut s = res.slice_mut(s![..1, ..]);
        s += &field.slice(s![1..2, ..]);
        s -= &field.slice(s![-1.., ..]);
    }
    {
        let mut s = res.slice_mut(s![-1.., ..]);
        s += &field.slice(s![..1, ..]);
        s -= &field.slice(s![-2..-1, ..]);
    }
    res
}

fn central_gradient_y(field: &Array2<f32>) -> Array2<f32> {
    let mut res: Array2<f32> = Array2::zeros(field.dim());

    // bulk
    {
        let mut s = res.slice_mut(s![.., 1..-1]);
        s += &field.slice(s![.., 2..]);
        s -= &field.slice(s![.., ..-2]);
    }
    // borders
    {
        let mut s = res.slice_mut(s![.., ..1]);
        s += &field.slice(s![.., 1..2]);
        s -= &field.slice(s![.., -1..]);
    }
    {
        let mut s = res.slice_mut(s![.., -1..]);
        s += &field.slice(s![.., ..1]);
        s -= &field.slice(s![.., -2..-1]);
    }
    res
}

pub fn array2_to_image(distance: &Array2<f32>) -> RgbaImage {
    let distance_without_inf = distance.mapv(|x| if x == INFINITY { 0.0 } else { x });

    let furthest = distance_without_inf.max().unwrap();

    let distance_pixels = distance
        .t()
        .iter()
        .flat_map(|d| {
            if *d == INFINITY {
                vec![255, 0, 255, 255]
            } else {
                let d_u8 = (d * 255.0 / furthest) as u8;
                vec![d_u8, d_u8, d_u8, 255]
            }
        })
        .collect();

    RgbaImage::from_raw(
        distance.nrows() as u32,
        distance.ncols() as u32,
        distance_pixels,
    )
    .unwrap()
}

pub(crate) fn array2_gradients_image(distance: &Array2<f32>) -> RgbaImage {
    let distance_without_inf = distance.mapv(|x| if x == INFINITY { 0.0 } else { x });

    let res_x = central_gradient_x(&distance_without_inf).mapv(|x| x.abs());
    let res_y = central_gradient_y(&distance_without_inf).mapv(|x| x.abs());

    RgbaImage::from_fn(
        distance.shape()[0] as u32,
        distance.shape()[1] as u32,
        |x, y| {
            Rgba([
                res_x[[x as usize, y as usize]] as u8,
                res_y[[x as usize, y as usize]] as u8,
                0,
                255,
            ])
        },
    )
}

pub fn array2_gradient_orientation_image(distance: &Array2<f32>) -> RgbaImage {
    let distance_without_inf = distance.mapv(|x| if x == INFINITY { 0.0 } else { x });

    let res_x = central_gradient_x(&distance_without_inf);
    let res_y = central_gradient_y(&distance_without_inf);

    RgbaImage::from_fn(
        distance.shape()[0] as u32,
        distance.shape()[1] as u32,
        |x, y| {
            let normalized = Vector2::new(
                res_x[[x as usize, y as usize]],
                res_y[[x as usize, y as usize]],
            )
            .normalize();

            Rgba([
                ((normalized.x + 1.0) * 127.5) as u8,
                ((normalized.y + 1.0) * 127.5) as u8,
                0,
                255,
            ])
        },
    )
}
